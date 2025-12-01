using UnityEngine;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Text;
using Debug = UnityEngine.Debug;
using System;
using System.Collections;
using System.Threading;
using TMPro;
using UnityEngine.UI;

public class PythonMeasurementProcessor : MonoBehaviour
{
    public GameObject loadingBox;

    public UIscriptAndy UIscriptAndy;
    // UI/INPUT REFERENCES 
    //[Header("User Input via Inspector")]
    public TMP_InputField userHeightInput;
    public TMP_InputField userAgeInput;
    // NOTE: userGenderInput string is now only used to initialize the dropdown caption, 
    // but the actual input is read directly from the dropdown object.
    public string userGenderInput = "Select Gender";

    public TMP_Dropdown Genderinput;

    [Header("UI Image Loaders")]
    public UIscriptAndy frontImageLoader;
    public UIscriptAndy sideImageLoader;

    // Reference to the script that will display the measurements
    [Header("Display Component Reference")]
    // Make sure the MeasurementDisplay script is assigned here in the Inspector!
    public MeasurementDisplay measurementDisplay;
    // ----------------------------------------------------------------------

    [Header("Python Configuration")]
    public string pythonPath = @"C:\Users\andre\miniconda3\envs\py312\python.exe";

    // 1. PROJECT ROOT
    public string script1RootPath = @"C:\Uni\MED7\Semester project\P7_UTRY_Py\Measurements_Calculation";

    // 2. PYTHON SEARCH PATH
    public string script1SrcPath = @"C:\Uni\MED7\Semester project\P7_UTRY_Py\Measurements_Calculation\body_measure\src";

    // 3. SCRIPT PATH (Not directly used in RunScript1Async, but kept for context)
    public string script1Path = @"C:\Uni\MED7\Semester project\P7_UTRY_Py\Measurements_Calculation\body_measure\src\body_measure\cli.py";


    [Header("Script 2 Files")]
    public string script2Path = @"C:\Uni\MED7\Semester project\P7_UTRY_Py\Andy.py";

    public string sizeChartCsvPath = @"C:\Uni\MED7\Semester project\P7_UTRY_Py\size_chart.csv";

    // BASE DATA FOLDER: The unique run folders will be created inside this path.
    public string baseDataFolderPath = "C:\\Uni\\MED7\\Semester project\\P7_UTRY_Py\\Data";

    public GameObject Panel;
    public GameObject frontImageText;
    public GameObject sideImageText;

    // Private variable for Coroutine management
    private Coroutine errorCoroutine;

    public GameObject greyPanel;

    // Data Structures
    [System.Serializable]
    public class MeasurementData
    {
        public float? height_cm;
        public float? chest_cm;
        public float? waist_cm;
        public float? hip_cm;
        public float? shoulder_width_cm;
    }

    [System.Serializable]
    public class JsonDictEntry
    {
        public string key;
        public float value;
    }

    [System.Serializable]
    public class PythonRuntimes
    {
        public float step_1_setup_ms;
        public float step_2_impute_ms;
        public float step_3_sizing_ms;
        public float step_4_export_ms;
        public float total_runtime_ms;
    }

    [System.Serializable]
    public class PythonOutputData
    {
        public string status;
        public string debug_message;
        public string recommended_size; // Holds the full output string (Size + Suffix + Comparison)
        public string scaled_measurements_json;
        public string final_measurements_json; // This holds the measurements JSON string
        public PythonRuntimes runtime_ms;
        public string simple_additional_info; // Holds the simple suggestion (e.g., "For a looser fit...")
    }

    void Start()
    {
        if (loadingBox != null)
            loadingBox.SetActive(false);
        //Errorbox.SetActive(false);

        Panel.gameObject.SetActive(false);
        frontImageText.gameObject.SetActive(true);
        sideImageText.gameObject.SetActive(true);
        greyPanel.gameObject.SetActive(true);
        // Ensure the dropdown caption reflects the default value at startup
        Genderinput.captionText.text = userGenderInput;
    }

    // Helper method to start the error display/timer
    private void ShowError(string message)
    {
        Debug.LogError(message);

        // Stop any running hide coroutine
        if (errorCoroutine != null)
        {
            StopCoroutine(errorCoroutine);
        }

        if (loadingBox != null) loadingBox.SetActive(false); // Turn off loading box on error
    }

    private string CleanPath(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return string.Empty;
        }

        char separator = Path.DirectorySeparatorChar;
        char altSeparator = Path.AltDirectorySeparatorChar;

        // Check for both Windows (\) and Unix (/) style separators
        if (path.EndsWith(separator.ToString()) || path.EndsWith(altSeparator.ToString()))
            return path.Substring(0, path.Length - 1);

        return path;
    }

    // Find the next available "Participant XX" folder name
    private string GetNextParticipantRunFolder()
    {
        // Ensure the base folder exists
        if (!Directory.Exists(baseDataFolderPath))
        {
            Debug.LogWarning($"Base data folder does not exist. Creating: {baseDataFolderPath}");
            Directory.CreateDirectory(baseDataFolderPath);
            return Path.Combine(baseDataFolderPath, "Participant 1");
        }

        // Scan for existing Participant folders
        string[] existingFolders = Directory.GetDirectories(baseDataFolderPath, "Participant *");
        int maxParticipantNumber = 0;

        // Use a natural language approach to parsing the folder names
        foreach (string folderPath in existingFolders)
        {
            string folderName = Path.GetFileName(folderPath);
            if (folderName.StartsWith("Participant "))
            {
                // Try to extract the number part
                string numberString = folderName.Substring("Participant ".Length);
                if (int.TryParse(numberString, out int currentNumber))
                {
                    if (currentNumber > maxParticipantNumber)
                    {
                        maxParticipantNumber = currentNumber;
                    }
                }
            }
        }

        // Determine the next number
        int nextParticipantNumber = maxParticipantNumber + 1;

        // Return the full path for the new folder
        string runID = $"Participant {nextParticipantNumber}";
        return Path.Combine(baseDataFolderPath, runID);
    }

    // 2. Button wrapper
    public void OnProcessButtonClicked()
    {
        Panel.gameObject.SetActive(true);
        frontImageText.gameObject.SetActive(false);
        sideImageText.gameObject.SetActive(false);

        // Check Image Loaders reference first
        if (frontImageLoader == null || sideImageLoader == null)
        {
            ShowError("Front or Side Image Loader is not assigned in the Inspector.");
            return;
        }

        string frontImagePath = frontImageLoader.selectedFilePath;
        string sideImagePath = sideImageLoader.selectedFilePath;

        // Validation 1: Check for selected images
        if (string.IsNullOrEmpty(frontImagePath) || string.IsNullOrEmpty(sideImagePath))
        {
            ShowError("Please select both front and side images first using the UI buttons.");
            return;
        }

        // Validation 2: Height
        float userHeight;
        if (!float.TryParse(userHeightInput.text, out userHeight))
        {
            ShowError("Please enter a valid numerical **HEIGHT** (e.g., 170.5).");
            return;
        }

        // Validation 3: Age
        float userAge;
        if (!float.TryParse(userAgeInput.text, out userAge))
        {
            ShowError("Please enter a valid numerical **AGE** (e.g., 25).");
            return;
        }

        // --- Validation 4: Gender (FIXED to read from Dropdown) ---
        float userGender = 0.0f;

        if (Genderinput == null)
        {
            ShowError("Gender Dropdown (Genderinput) is not assigned in the Inspector.");
            return;
        }

        // Get the text of the currently selected option using captionText
        string genderInputText = Genderinput.captionText.text;
        string genderInput = genderInputText.ToLower().Trim();

        if (string.IsNullOrWhiteSpace(genderInput) || genderInput == "select gender")
        {
            ShowError("Please select a valid **GENDER** from the dropdown.");
            return;
        }

        if (genderInput == "male")
        {
            userGender = 1.0f;
        }
        else if (genderInput == "female")
        {
            userGender = 0.1f;
        }
        else if (genderInput == "non-binary" || genderInput == "nonbinary")
        {
            userGender = 2.0f;
        }
        else
        {
            ShowError($"Invalid Gender input detected: '{genderInputText}'. Dropdown value must be 'Male', 'Female', or 'Nonbinary'.");
            return;
        }
        // -----------------------------------------------------------

        // --- UPDATED: Generate Unique Run Folder Path ---
        // Get the next available participant folder path
        string runDataFolderPath = GetNextParticipantRunFolder();
        // -------------------------------------------------

        CheckPythonDependencies();

        // Pass the new unique folder path to the main process
        StartFullProcess(userHeight, userAge, userGender, frontImagePath, sideImagePath, runDataFolderPath);
    }

    // --- Dependency Check ---
    private void CheckPythonDependencies()
    {
        string[] requiredPackages = { "pandas", "joblib" };
        string baseCommand = $"{pythonPath} -c \"import {requiredPackages[0]}; import {requiredPackages[1]}; print('Dependencies OK')\"";

        ProcessStartInfo start = new ProcessStartInfo
        {
            FileName = "cmd.exe",
            Arguments = $"/C {baseCommand}",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };

        try
        {
            using (Process process = Process.Start(start))
            {
                string output = process.StandardOutput.ReadToEnd();
                string error = process.StandardError.ReadToEnd();
                process.WaitForExit(5000);

                if (process.ExitCode != 0 || !output.Contains("Dependencies OK"))
                {
                    Debug.LogError($"PYTHON ENVIRONMENT WARNING: Kunne ikke importere 'pandas' eller 'joblib' i din Python-omgivelse ({Path.GetDirectoryName(pythonPath)}). Fejl: {error}");
                    Debug.LogError("Dette er sandsynligvis årsagen til Code 1-fejlen. Kør: 'pip install pandas joblib' i din py312 conda-omgivelse.");
                }
                else
                {
                    Debug.Log("✅ Python afhængigheder (Pandas, Joblib) er installeret i din conda-omgivelse.");
                }
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"❌ Fejl ved check af Python-afhængigheder: {e.Message}");
        }
    }


    // MAIN ASYNC PIPELINE 
    public async void StartFullProcess(float userHeight, float userAge, float userGender, string frontImagePath, string sideImagePath, string runDataFolderPath)
    {
        loadingBox.gameObject.SetActive(true);

        Debug.Log($"Starting Python processing pipeline for Run ID: {Path.GetFileName(runDataFolderPath)}");

        string dataFolderFullPath = Path.GetFullPath(runDataFolderPath);
        if (!Directory.Exists(dataFolderFullPath))
        {
            Directory.CreateDirectory(dataFolderFullPath);
            Debug.Log($"Created unique data folder: {dataFolderFullPath}");
        }

        // --- TRIN 1: KØR SCRIPT 1 (Measurement Calculation) ---
        Debug.Log("Trin 1/2: Kører Script 1 for måling...");

        System.Diagnostics.Stopwatch script1Timer = new System.Diagnostics.Stopwatch();
        script1Timer.Start();

        // Pass the unique runDataFolderPath for output
        string script1JsonOutput = await RunScript1Async(userHeight, frontImagePath, sideImagePath, script1RootPath, dataFolderFullPath);

        script1Timer.Stop();
        float script1TotalRuntimeMS = (float)script1Timer.Elapsed.TotalMilliseconds;
        float script1TotalRuntimeS = script1TotalRuntimeMS / 1000.0f;

        Debug.Log($"✅ Script 1 (Image Measurement) Total Runtime: {script1TotalRuntimeMS:F1} ms ({script1TotalRuntimeS:F3} s)");

        if (string.IsNullOrEmpty(script1JsonOutput))
        {
            ShowError("Pipeline Error: Script 1 failed to return JSON output. Stopping.");
            return;
        }

        // Use the runDataFolderPath to save the input file for script 2
        string inputForScript2Path = Path.GetFullPath(Path.Combine(runDataFolderPath, "input_for_imputation.json"));
        File.WriteAllText(inputForScript2Path, script1JsonOutput);
        Debug.Log($"📝 JSON Output fra Script 1 gemt som input til Script 2: {inputForScript2Path}");


        // --- TRIN 2: KØR SCRIPT 2 (Imputation & Sizing) ---
        Debug.Log("Trin 2/2: Kører Script 2 for imputation og størrelse...");

        // PASS THE NEW ARGUMENT HERE: runDataFolderPath
        string finalJsonOutput = await RunScript2Async(inputForScript2Path, userHeight, userAge, userGender, runDataFolderPath);

        if (!string.IsNullOrEmpty(finalJsonOutput))
        {
            try
            {
                PythonOutputData data = JsonUtility.FromJson<PythonOutputData>(finalJsonOutput);

                if (data.status == "error")
                {
                    string errorMessage = data.recommended_size ?? "Unknown error in Python.";
                    string rawErrorData = data.final_measurements_json ?? "No raw data.";
                    ShowError($"❌ Python Script 2 ({Path.GetFileName(script2Path)}) returnerede fejl (Status=error): {errorMessage}. Rå fejl-data: {rawErrorData}");
                    return;
                }

                Debug.Log($"👍 BEHANDLING FULDFØRT!");

                // --- PASS THE NEW IMAGE PATH (using the unique path) ---
                string frontOverlayPath = Path.GetFullPath(Path.Combine(runDataFolderPath, "front_overlay.png"));


                // =========================================================================================
                // ⭐ START: FINAL PARSING LOGIC FOR TRIPLE OUTPUT
                // =========================================================================================

                string simpleSize = "N/A";
                string simpleFitDescription = "No fit information.";
                string fullDetailedComparison = "No detailed comparison available.";

                if (!string.IsNullOrEmpty(data.recommended_size))
                {
                    string fullRecommendedString = data.recommended_size.Trim();
                    string[] lines = fullRecommendedString.Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);

                    string rawFirstLine = lines.Length > 0 ? lines[0].Trim() : "";

                    // --- 1. Extract Recommended Size (Simple Size) ---
                    if (rawFirstLine.Contains("Recommended:"))
                    {
                        string sizeAndSuffix = rawFirstLine.Replace("Recommended:", "").Trim();
                        int suffixIndex = sizeAndSuffix.IndexOf(" (");

                        if (suffixIndex > 0)
                        {
                            // Size found (e.g., "M")
                            simpleSize = sizeAndSuffix.Substring(0, suffixIndex).Trim();
                        }
                        else
                        {
                            // Only size found (no tight/loose advice)
                            simpleSize = sizeAndSuffix;
                        }
                    }
                    else if (fullRecommendedString.StartsWith("ERROR:"))
                    {
                        simpleSize = "ERROR";
                        simpleFitDescription = fullRecommendedString;
                    }

                    // --- 2. Extract Simple Fit Description (e.g., "Can have bit loose fit") ---
                    // This is the part inside the parenthesis: (Can have bit loose fit)
                    int openParenIndex = fullRecommendedString.IndexOf('(');
                    int closeParenIndex = fullRecommendedString.IndexOf(')');

                    if (openParenIndex != -1 && closeParenIndex != -1 && closeParenIndex > openParenIndex)
                    {
                        // Extract the content *including* the parentheses
                        simpleFitDescription = fullRecommendedString.Substring(openParenIndex, closeParenIndex - openParenIndex + 1).Trim();

                        // Apply custom capitalization for the suffix (e.g., "(Can...")
                        if (simpleFitDescription.Length > 1 && simpleFitDescription.StartsWith("("))
                        {
                            int capIndex = -1;
                            for (int i = 1; i < simpleFitDescription.Length; i++)
                            {
                                if (char.IsLetter(simpleFitDescription[i]))
                                {
                                    capIndex = i;
                                    break;
                                }
                            }

                            if (capIndex != -1)
                            {
                                char firstChar = simpleFitDescription[capIndex];
                                if (char.IsLower(firstChar))
                                {
                                    simpleFitDescription = simpleFitDescription.Substring(0, capIndex) + char.ToUpper(firstChar) + simpleFitDescription.Substring(capIndex + 1);
                                }
                            }
                        }
                    }
                    else
                    {
                        simpleFitDescription = $"Size {simpleSize} is the best fit.";
                    }

                    // --- 3. Extract Full Detailed Comparison (Pop-up Content) ---
                    // The detailed comparison starts with 'Compared with the next best size'
                    int comparisonHeaderIndex = fullRecommendedString.IndexOf("Compared with the next best size", 0, StringComparison.Ordinal);

                    if (comparisonHeaderIndex != -1)
                    {
                        // Start extracting from the header (including the header)
                        fullDetailedComparison = fullRecommendedString.Substring(comparisonHeaderIndex);

                        // Clean up the comparison block: remove the HTML-style underline tags
                        fullDetailedComparison = fullDetailedComparison.Replace("<u>", "").Replace("</u>", "").Trim();
                    }
                    else
                    {
                        // If no comparison block is found, use a default message
                        fullDetailedComparison = "No detailed comparison available.";
                    }
                }

                // Append the simple suggestion line from Python (data.simple_additional_info) to the fit description
                if (!string.IsNullOrEmpty(data.simple_additional_info))
                {
                    // Use the simple_additional_info from Python if it exists, to replace the default message
                    // Format: "(Can have bit loose fit)\nNext best size is M"
                    simpleFitDescription = $"{simpleFitDescription}\n{data.simple_additional_info}";
                }

                // FINAL CHECK: If comparison block is missing but we have a simple suggestion, use the simple one
                if (fullDetailedComparison == "No detailed comparison available." && !string.IsNullOrEmpty(data.simple_additional_info))
                {
                    fullDetailedComparison = data.simple_additional_info;
                }


                // =========================================================================================
                // ⭐ END: FINAL PARSING LOGIC
                // =========================================================================================

                if (measurementDisplay != null && !string.IsNullOrEmpty(data.final_measurements_json))
                {
                    // Pass the three extracted strings to the display component
                    measurementDisplay.DisplayMeasurementsFromPython(
                        data.final_measurements_json,        // Argument 1: Measurement JSON
                        simpleSize,                            // Argument 2: Simple Size (e.g., "M")
                        frontOverlayPath,                    // Argument 3: Image Path
                        simpleFitDescription,                // Argument 4: Fit Suffix + Simple Suggestion
                        fullDetailedComparison               // Argument 5: Full Pop-up Comparison Text
                    );
                    Debug.Log("Measurements, size, and image path successfully passed to the display component.");
                }
                else if (measurementDisplay == null)
                {
                    Debug.LogError("Measurement Display component (MeasurementDisplay script) is not assigned in the Inspector!");
                }
                else
                {
                    Debug.LogError("Python output 'final_measurements_json' was empty. Cannot display results.");
                }

                // --- turn loading box OFF here (right after UI updates) ---
                if (loadingBox != null)
                    loadingBox.SetActive(false);

                if (greyPanel != null)
                    greyPanel.SetActive(false);

                // --- KØRSELSTID STATISTIK ---
                Debug.Log("--- TOTAL KØRSELSTID STATISTIK ---");
                Debug.Log($"Script 1 (Image Measurement) Total: {script1TotalRuntimeMS:F1} ms ({script1TotalRuntimeS:F3} s)");

                Debug.Log($"Script 2 TOTAL Internal: {data.runtime_ms.total_runtime_ms:F1} ms ({(data.runtime_ms.total_runtime_ms / 1000.0f):F6} s)");

                float overallTotalRuntimeMS = script1TotalRuntimeMS + data.runtime_ms.total_runtime_ms;
                float overallTotalRuntimeS = overallTotalRuntimeMS / 1000.0f;

                Debug.Log($"TOTAL TIME : {overallTotalRuntimeMS:F1} ms ({overallTotalRuntimeS:F3} s)");
            }
            catch (System.Exception e)
            {
                ShowError($"JSON Parsing Error: Kunne ikke deserialisere output fra Script 2. Rå output: {finalJsonOutput}. Fejl: {e.Message}");
            }
        }
        else
        {
            ShowError("Pipeline Error: Script 2 fejlede eller returnerede intet output.");
        }
    }

    // ----------------------------------------------------------------------
    // --- Execution Methods
    // ----------------------------------------------------------------------

    private Task<string> RunScript1Async(float heightCm, string frontImgPath, string sideImgPath, string pythonRootPath, string outputDirPath)
    {
        // FIX: Use the CleanPath helper method.
        string cleanedOutputPath = CleanPath(outputDirPath);

        return Task.Run(() =>
        {
            string arguments = $"-m body_measure.cli " +
                                $"--front \"{frontImgPath}\" " +
                                $"--side \"{sideImgPath}\" " +
                                $"--height-cm {heightCm} " +
                                $"--backend deeplabv3 " +
                                $"--device cpu " +
                                $"--debug-dir \"{cleanedOutputPath}\" " + // Using the unique run path
                                $" --save-masks";

            return ExecutePythonProcess(arguments, "body_measure.cli", pythonRootPath, script1SrcPath);
        });
    }

    private Task<string> RunScript2Async(string inputFilePath, float userHeight, float userAge, float userGender, string runDataFolderPath)
    {
        // Add GetFullPath here for maximum certainty on the path string integrity
        string absoluteSizeChartPath = Path.GetFullPath(sizeChartCsvPath);

        return Task.Run(() =>
        {
            // Pass userHeight as 3rd arg, userAge as 4th arg, userGender as 5th arg, outputFolder as 6th arg
            string arguments = $"\"{script2Path}\" \"{inputFilePath}\" \"{absoluteSizeChartPath}\" \"{userHeight}\" \"{userAge}\" \"{userGender}\" \"{runDataFolderPath}\"";

            string script2WorkingDir = Path.GetDirectoryName(script2Path);

            return ExecutePythonProcess(arguments, Path.GetFileName(script2Path), script2WorkingDir, null);
        });
    }

    // ----------------------------------------------------------------------
    // --- CORE EXECUTION METHOD ---
    // ----------------------------------------------------------------------
    private string ExecutePythonProcess(string arguments, string scriptIdentifier, string workingDirectory, string pythonSrcPath)
    {
        ProcessStartInfo start = new ProcessStartInfo
        {
            FileName = pythonPath,
            Arguments = arguments,
            WorkingDirectory = workingDirectory,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };

        if (!string.IsNullOrEmpty(pythonSrcPath))
        {
            start.EnvironmentVariables["PYTHONPATH"] = pythonSrcPath;
            Debug.Log($"Setting PYTHONPATH to: {pythonSrcPath}");
        }

        start.EnvironmentVariables["KMP_DUPLICATE_LIB_OK"] = "TRUE";
        start.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";

        try
        {
            using (Process process = new Process())
            {
                process.StartInfo = start;
                process.EnableRaisingEvents = true;

                StringBuilder output = new StringBuilder();
                StringBuilder error = new StringBuilder();

                using (ManualResetEvent processExited = new ManualResetEvent(false))
                {
                    process.OutputDataReceived += (sender, e) => {
                        if (e.Data != null) output.AppendLine(e.Data);
                    };

                    process.ErrorDataReceived += (sender, e) => {
                        if (e.Data != null) error.AppendLine(e.Data);
                    };

                    process.Exited += (sender, e) => {
                        processExited.Set();
                    };

                    process.Start();

                    process.BeginOutputReadLine();
                    process.BeginErrorReadLine();

                    process.WaitForExit();
                    processExited.WaitOne();

                    process.CancelOutputRead();
                    process.CancelErrorRead();

                    string outputString = output.ToString().Trim();
                    string errorString = error.ToString().Trim();
                    int exitCode = process.ExitCode;

                    if (exitCode != 0)
                    {
                        // Log error, but return outputString as it might contain Python's error JSON
                        Debug.LogError($"❌ Python Script '{scriptIdentifier}' Failed (Code {exitCode}): {errorString}");
                        Debug.LogError($"Executed command: {pythonPath} {arguments}");
                        Debug.LogError($"Working Directory: {workingDirectory}");
                        return outputString;
                    }

                    Debug.Log($"Python Script '{scriptIdentifier}' Success. (Output length: {outputString.Length})");
                    return outputString;
                }
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"❌ Failed to execute Python process: {e.Message}");
            return null;
        }
    }
}