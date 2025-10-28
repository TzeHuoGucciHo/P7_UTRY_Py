using UnityEngine;
using System.Diagnostics;
using System.IO;

public class PythonRunner : MonoBehaviour
{
    public string pythonPath = "C:/Users/marku/AppData/Local/Programs/Python/Python312/python.exe";
    public string pythonScriptPath = "Insert own path here";
    //@"C:\Users\marku\OneDrive - Aalborg Universitet\Githubs\P7_UTRY_Py\test.py";

    public void RunPythonScript()
    {
        ProcessStartInfo start = new ProcessStartInfo();
        start.FileName = pythonPath;
        start.Arguments = $"\"{pythonScriptPath}\"";
        start.WorkingDirectory = Path.GetDirectoryName(pythonScriptPath);
        start.UseShellExecute = false;
        start.RedirectStandardOutput = true;
        start.RedirectStandardError = true;
        start.CreateNoWindow = true;

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
