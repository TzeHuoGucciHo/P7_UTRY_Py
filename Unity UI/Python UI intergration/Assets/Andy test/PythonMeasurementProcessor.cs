using UnityEngine;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Text;
using Debug = UnityEngine.Debug;
using System;
using System.Threading;
using TMPro;
using UnityEngine.UI;

public class PythonMeasurementProcessor : MonoBehaviour
{
    public GameObject loadingBox;

    public UIscriptAndy UIscriptAndy;
    // --- UI/INPUT REFERENCES ---
    //[Header("User Input via Inspector")]
    public TMP_InputField userHeightInput;
    public TMP_InputField userAgeInput;
    public TMP_InputField userGenderInput;

    [Header("UI Image Loaders")]
    public UIscriptAndy frontImageLoader;
    public UIscriptAndy sideImageLoader;

    // --- NEW: Reference to the script that will display the measurements ---
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

    // --- Data Structures ---
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
        public string recommended_size;
        public string scaled_measurements_json;
        public string final_measurements_json; // This holds the measurements JSON string
        public PythonRuntimes runtime_ms;
    }

    void Start()
    {
        if (loadingBox != null)
            loadingBox.SetActive(false);
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

    // =================================================================================
    // 2. BUTTON WRAPPER
    // =================================================================================
    public void OnProcessButtonClicked()
    {
        loadingBox.gameObject.SetActive(true);

        UIscriptAndy.Panel.SetActive(true);
        // --- Input Validation (Height, Age, Gender) ---
        float userHeight;
        if (!float.TryParse(userHeightInput.text, out userHeight))
        {
            Debug.LogError("Validation Error: Please enter a valid numerical **HEIGHT** in the Inspector field (e.g., 170.5).");
            return;
        }

        float userAge;
        if (!float.TryParse(userAgeInput.text, out userAge))
        {
            Debug.LogError("Validation Error: Please enter a valid numerical **AGE** in the Inspector field (e.g., 25).");
            return;
        }

        // --- Gender Validation ---
        float userGender = 0.0f;
        string gender = userGenderInput.text.ToLower().Trim();

        if (gender == "male")
        {
            userGender = 1.0f;
        }
        else if (gender == "female")
        {
            userGender = 0.0f;
        }
        else if (gender == "non-binary" || gender == "nonbinary")
        {
            userGender = 2.0f;
        }

        if (frontImageLoader == null || sideImageLoader == null)
        {
            Debug.LogError("Reference Error: Front or Side Image Loader is not assigned in the Inspector.");
            return;
        }

        string frontImagePath = frontImageLoader.selectedFilePath;
        string sideImagePath = sideImageLoader.selectedFilePath;

        if (string.IsNullOrEmpty(frontImagePath) || string.IsNullOrEmpty(sideImagePath))
        {
            Debug.LogError("Validation Error: Please select both front and side images first using the UI buttons.");
            return;
        }

        // --- NEW: Generate Unique Run ID and Folder Path ---
        string runID = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        string runDataFolderPath = Path.Combine(baseDataFolderPath, runID);
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


    // =================================================================================
    // 3. MAIN ASYNC PIPELINE (UPDATED to accept runDataFolderPath)
    // =================================================================================
    public async void StartFullProcess(float userHeight, float userAge, float userGender, string frontImagePath, string sideImagePath, string runDataFolderPath)
    {
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
            Debug.LogError("Pipeline Error: Script 1 failed to return JSON output. Stopping.");
            if (loadingBox != null) loadingBox.SetActive(false);
            return;
        }

        // Use the runDataFolderPath to save the input file for script 2
        string inputForScript2Path = Path.GetFullPath(Path.Combine(runDataFolderPath, "input_for_imputation.json"));
        File.WriteAllText(inputForScript2Path, script1JsonOutput);
        Debug.Log($"📝 JSON Output fra Script 1 gemt som input til Script 2: {inputForScript2Path}");


        // --- TRIN 2: KØR SCRIPT 2 (Imputation & Sizing) ---
        Debug.Log("Trin 2/2: Kører Script 2 for imputation og størrelse...");

        string finalJsonOutput = await RunScript2Async(inputForScript2Path, userHeight, userAge, userGender);

        if (!string.IsNullOrEmpty(finalJsonOutput))
        {
            try
            {
                PythonOutputData data = JsonUtility.FromJson<PythonOutputData>(finalJsonOutput);

                if (data.status == "error")
                {
                    string errorMessage = data.recommended_size ?? "Unknown error in Python.";
                    string rawErrorData = data.final_measurements_json ?? "No raw data.";
                    Debug.LogError($"❌ Python Script 2 ({Path.GetFileName(script2Path)}) returnerede fejl (Status=error): {errorMessage}. Rå fejl-data: {rawErrorData}");
                    if (loadingBox != null) loadingBox.SetActive(false);
                    return;
                }

                Debug.Log($"👍 BEHANDLING FULDFØRT!");
                Debug.Log($"📝 Python Debug Info: {data.debug_message}");
                Debug.Log($"📏 Anbefalet størrelse: **{data.recommended_size}**");
                Debug.Log($"Imputerede mål (JSON String): {data.final_measurements_json}");

                // --- PASS THE NEW IMAGE PATH (using the unique path) ---
                string frontOverlayPath = Path.GetFullPath(Path.Combine(runDataFolderPath, "front_overlay.png"));

                if (measurementDisplay != null && !string.IsNullOrEmpty(data.final_measurements_json))
                {
                    measurementDisplay.DisplayMeasurementsFromPython(
                        data.final_measurements_json, // Argument 1: The measurement data
                        data.recommended_size,       // Argument 2: The recommended size/info string
                        frontOverlayPath             // Argument 3: The path to the overlay image
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
                Debug.LogError($"JSON Parsing Error: Kunne ikke deserialisere output fra Script 2. Rå output: {finalJsonOutput}. Fejl: {e.Message}");
                if (loadingBox != null) loadingBox.SetActive(false);
            }
        }
        else
        {
            Debug.LogError("Pipeline Error: Script 2 fejlede eller returnerede intet output.");
            if (loadingBox != null) loadingBox.SetActive(false);
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

    private Task<string> RunScript2Async(string inputFilePath, float userHeight, float userAge, float userGender)
    {
        // Add GetFullPath here for maximum certainty on the path string integrity
        string absoluteSizeChartPath = Path.GetFullPath(sizeChartCsvPath);

        return Task.Run(() =>
        {
            // Pass userHeight as 3rd arg, userAge as 4th arg, userGender as 5th arg
            string arguments = $"\"{script2Path}\" \"{inputFilePath}\" \"{absoluteSizeChartPath}\" \"{userHeight}\" \"{userAge}\" \"{userGender}\"";

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
