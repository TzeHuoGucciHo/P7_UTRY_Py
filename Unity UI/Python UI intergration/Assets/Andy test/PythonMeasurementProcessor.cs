using UnityEngine;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Text;
using Debug = UnityEngine.Debug;

public class PythonMeasurementProcessor : MonoBehaviour
{
    // --- UI/INPUT REFERENCES (Unchanged) ---
    [Header("User Input via Inspector")]
    public string userHeightString = "190.0";

    [Header("UI Image Loaders")]
    public UIscriptAndy frontImageLoader;
    public UIscriptAndy sideImageLoader;

    [Header("Python Configuration")]
    public string pythonPath = @"C:\Users\andre\AppData\Local\Programs\Python\Python312\python.exe";

    // 1. PROJECT ROOT: The folder containing 'body_measure' and 'cli.py' relative parent folders.
    public string script1RootPath = @"C:\Uni\MED7\Semester project\P7_UTRY_Py\Measurements_Calculation";

    // 2. PYTHON SEARCH PATH (PYTHONPATH): The folder containing the 'body_measure' package folder.
    // This is necessary for internal imports like 'from body_measure import measure_v2'.
    public string script1SrcPath = @"C:\Uni\MED7\Semester project\P7_UTRY_Py\Measurements_Calculation\body_measure\src";

    // 3. SCRIPT PATH: The exact location of the script being executed.
    public string script1Path = @"C:\Uni\MED7\Semester project\P7_UTRY_Py\Measurements_Calculation\body_measure\src\body_measure\cli.py";


    public string script2Path = @"C:\Uni\MED7\Semester project\P7_UTRY_Py\Andy test.py";
    public string dataFolderPath = "Data/";

    [Header("Script 2 Data File")]
    public string script2DataFilePath = @"C:\PATH\TO\YOUR\Body Measurements _ original_CSV.csv";

    // --- Data Structures (Unchanged) ---
    [System.Serializable]
    public class MeasurementData
    {
        public float? height_cm;
        public float? chest_cm;
        public float? waist_cm;
        public float? hip_cm;
        public float? shoulder_width_cm;
        public string recommended_size;
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
        // Must match the key names in the Python JSON output
        public string status;
        public string file_saved;
        public string path;
        public PythonRuntimes runtime_ms;
    }
    // =================================================================================


    // =================================================================================
    // 1. BUTTON WRAPPER (Unchanged)
    // =================================================================================
    public void OnProcessButtonClicked()
    {
        Debug.Log("Button clicked. Starting data validation...");

        float userHeight;
        if (!float.TryParse(userHeightString, out userHeight))
        {
            Debug.LogError("Validation Error: Please enter a valid numerical height in the Inspector field (e.g., 170.5).");
            return;
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

        MeasurementData knownMeasurements = new MeasurementData()
        {
            chest_cm = 103.5f,
            waist_cm = 77f,
            hip_cm = 100f
        };

        StartFullProcess(userHeight, frontImagePath, sideImagePath, knownMeasurements);
    }

    // =================================================================================
    // 2. MAIN ASYNC PIPELINE (Unchanged)
    // =================================================================================
    public async void StartFullProcess(float userHeight, string frontImagePath, string sideImagePath, MeasurementData knownMeasurements)
    {
        Debug.Log("Starting Python processing pipeline...");

        // Ensure the data output folder exists
        string dataFolderFullPath = Path.GetFullPath(dataFolderPath);
        if (!Directory.Exists(dataFolderFullPath))
        {
            Directory.CreateDirectory(dataFolderFullPath);
        }

        // We use dataFolderFullPath as the debug-dir and out-json output folder
        string script1JsonOutput = await RunScript1Async(userHeight, frontImagePath, sideImagePath, script1RootPath, dataFolderFullPath);

        if (string.IsNullOrEmpty(script1JsonOutput))
        {
            Debug.LogError("Pipeline Error: Script 1 failed to return JSON output. Stopping.");
            return;
        }

        string inputForScript2Path = Path.GetFullPath(Path.Combine(dataFolderPath, "input_for_imputation.json")); // Resolve to absolute path
        File.WriteAllText(inputForScript2Path, script1JsonOutput);

        Debug.Log($"📝 Script 2 input JSON saved to: {inputForScript2Path}"); // Log the absolute path

        string finalJsonOutput = await RunScript2Async(inputForScript2Path);

        if (!string.IsNullOrEmpty(finalJsonOutput))
        {
            try
            {
                PythonOutputData data = JsonUtility.FromJson<PythonOutputData>(finalJsonOutput);

                Debug.Log($"✅ Processing Complete. File saved to: {data.path}");

                // Log the runtime details
                Debug.Log("--- RUNTIME STATISTICS ---");
                Debug.Log($"Setup/Load (ms): {data.runtime_ms.step_1_setup_ms:F1}");
                Debug.Log($"Imputation (ms): {data.runtime_ms.step_2_impute_ms:F1}");
                Debug.Log($"Sizing Logic (ms): {data.runtime_ms.step_3_sizing_ms:F1}");
                Debug.Log($"Export JSON (ms): {data.runtime_ms.step_4_export_ms:F1}");
                Debug.Log($"TOTAL Runtime (ms): {data.runtime_ms.total_runtime_ms:F1}");
                Debug.Log("--------------------------");
            }
            catch (System.Exception e)
            {
                Debug.LogError($"JSON Parsing Error: Could not deserialize runtime data. Raw output: {finalJsonOutput}. Error: {e.Message}");
            }
        }
        else
        {
            Debug.LogError("Pipeline Error: Script 2 failed to return final JSON output.");
        }
    }

    // ----------------------------------------------------------------------
    // --- Execution Methods ---
    // ----------------------------------------------------------------------

    // 🚀 RESTORED: Passing script1SrcPath to set PYTHONPATH.
    private Task<string> RunScript1Async(float heightCm, string frontImgPath, string sideImgPath, string pythonRootPath, string outputDirPath)
    {
        return Task.Run(() =>
        {
            // *** CRITICAL FIX: Changed --backend deeplabv3 to --backend opencv ***
            // DeepLabV3 requires PyTorch, which is not installed. Using 'opencv' forces the HOG/GrabCut fallback.
            string arguments = $"\"{script1Path}\" " +
                               $"--front \"{frontImgPath}\" " +
                               $"--side \"{sideImgPath}\" " +
                               $"--height-cm {heightCm} " +
                               $"--backend opencv " +
                               $"--device cpu " +
                               $"--debug-dir \"{outputDirPath}\" " +
                               $"--save-masks " +
                               $"--out-json \"{outputDirPath}\\output_measurements.json\"";

            // Pass the source path to set PYTHONPATH so internal imports resolve.
            return ExecutePythonProcess(arguments, Path.GetFileName(script1Path), pythonRootPath, script1SrcPath);
        });
    }

    private Task<string> RunScript2Async(string inputFilePath)
    {
        return Task.Run(() =>
        {
            string arguments = $"\"{script2Path}\" \"{inputFilePath}\" \"{script2DataFilePath}\"";
            return ExecutePythonProcess(arguments, Path.GetFileName(script2Path), Path.GetDirectoryName(script2Path), null);
        });
    }

    private string ExecutePythonProcess(string arguments, string scriptIdentifier, string workingDirectory, string pythonSrcPath)
    {
        ProcessStartInfo start = new ProcessStartInfo
        {
            FileName = pythonPath,
            Arguments = arguments,
            WorkingDirectory = workingDirectory, // The folder containing the 'cli.py' file (or the root if cli.py calls modules)
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,

            // C# side encoding
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };

        // 💡 CRITICAL FIX: Set PYTHONPATH if a source path is provided.
        if (!string.IsNullOrEmpty(pythonSrcPath))
        {
            // This tells the Python interpreter where to look for the 'body_measure' package.
            start.EnvironmentVariables["PYTHONPATH"] = pythonSrcPath;
            Debug.Log($"Setting PYTHONPATH to: {pythonSrcPath}");
        }

        // Force Python to use UTF-8 for its output streams.
        start.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";

        try
        {
            using (Process process = Process.Start(start))
            {
                string output = process.StandardOutput.ReadToEnd();
                string error = process.StandardError.ReadToEnd();

                process.WaitForExit();

                if (process.ExitCode != 0)
                {
                    Debug.LogError($"❌ Python Script '{scriptIdentifier}' Failed (Code {process.ExitCode}): {error}");
                    Debug.LogError($"Executed command: {pythonPath} {arguments}");
                    Debug.LogError($"Working Directory: {workingDirectory}");
                    return null;
                }

                Debug.Log($"Python Script '{scriptIdentifier}' Success. (Output length: {output.Length})");
                return output.Trim();
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"❌ Failed to execute Python process: {e.Message}");
            return null;
        }
    }
}