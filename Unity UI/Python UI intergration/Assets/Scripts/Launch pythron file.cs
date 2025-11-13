using UnityEngine;
using System.Diagnostics;
using System.IO;

public class PythonRunner : MonoBehaviour
{


    public string pythonPath = "cmd.exe";
    public string pythonScriptPath = @"C:\Users\marku\OneDrive - Aalborg Universitet\Githubs\P7_UTRY_Py\Measurements_Calculation\body_measure";

    public string pythonCommand = @"python -m body_measure.cli --front data\L_front2.jpg --side data\L_side2.jpg --height-cm 190 --backend deeplabv3 --device cuda --debug-dir .\debug_single --save-masks --out-json .\debug_single\Lukas_190cm_results.json";
    //@"C:\Users\marku\OneDrive - Aalborg Universitet\Githubs\P7_UTRY_Py\test.py";

    public void RunPythonScript()
    {
        ProcessStartInfo start = new ProcessStartInfo
        {
            FileName = pythonPath,
            Arguments = $"/K {pythonCommand}",
            WorkingDirectory = Path.GetDirectoryName(pythonScriptPath),
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };
        
        

        using (Process process = Process.Start(start))
        {
            string output = process.StandardOutput.ReadToEnd();
            string error = process.StandardError.ReadToEnd();
            process.WaitForExit();

            UnityEngine.Debug.Log("Python Output: " + output);
            if (!string.IsNullOrEmpty(error))
                UnityEngine.Debug.LogError("Python Error: " + error);
        }
    }
}
