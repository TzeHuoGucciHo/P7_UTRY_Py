using System.Diagnostics;
using UnityEngine;
using UnityEngine.UI;
using System;
using System.IO;
using Debug = UnityEngine.Debug;

public class PythonRunner : MonoBehaviour
{
    public RawImage imageDisplay;          // Assign in inspector
    public string pythonExePath;           // Path to python.exe
    public string scriptPath;              // The python script
    public string outputImagePath = "cropped_output.png";

    // --- ADDED: This is where the path will be stored, similar to UIscriptAndy's variable.
    public string selectedFilePath = "";

    public void RunPythonScript()
    {
        // 1. Basic Validation
        if (string.IsNullOrEmpty(selectedFilePath) || !File.Exists(selectedFilePath))
        {
            Debug.LogError("Input file path is empty or file does not exist. Please select an image first.");
            return;
        }

        ProcessStartInfo psi = new ProcessStartInfo();
        psi.FileName = pythonExePath;

        // --- MODIFIED: Pass the input and output file paths as arguments ---
        // Format: "script.py" "input.png" "output.png"
        string arguments = $"\"{scriptPath}\" \"{selectedFilePath}\" \"{outputImagePath}\"";

        psi.Arguments = arguments;
        Debug.Log($"Executing Python: {psi.FileName} {psi.Arguments}");

        psi.CreateNoWindow = true;
        psi.UseShellExecute = false;
        psi.RedirectStandardOutput = true;
        psi.RedirectStandardError = true;

        Process process = new Process();
        process.StartInfo = psi;

        try
        {
            process.Start();
            process.WaitForExit(); // 🛑 WARNING: This line will block Unity.

            // Read output and error after the process closes
            string output = process.StandardOutput.ReadToEnd();
            string error = process.StandardError.ReadToEnd();

            if (process.ExitCode != 0)
            {
                Debug.LogError($"❌ Python Script Failed (Code {process.ExitCode}): {error}");
                return;
            }

            Debug.Log($"Python Output: {output}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to start Python process. Error: {e.Message}");
            return;
        }

        LoadCroppedImage();
    }

    void LoadCroppedImage()
    {
        if (!File.Exists(outputImagePath))
        {
            Debug.LogError($"Output file not found at: {outputImagePath}");
            return;
        }

        byte[] bytes = File.ReadAllBytes(outputImagePath);
        Texture2D tex = new Texture2D(2, 2);
        tex.LoadImage(bytes);

        imageDisplay.texture = tex;
        FitRawImageToParent(imageDisplay, tex);
    }

    void FitRawImageToParent(RawImage raw, Texture2D tex)
    {
        RectTransform rt = raw.rectTransform;
        RectTransform parent = raw.transform.parent.GetComponent<RectTransform>();

        if (parent == null) return; // Exit if no parent RectTransform

        float texW = tex.width;
        float texH = tex.height;

        float parentW = parent.rect.width;
        float parentH = parent.rect.height;

        float texAspect = texW / texH;
        float parentAspect = parentW / parentH;

        float finalW, finalH;

        if (texAspect > parentAspect)
        {
            // Fit by width
            finalW = parentW;
            finalH = finalW / texAspect;
        }
        else
        {
            // Fit by height
            finalH = parentH;
            finalW = finalH * texAspect;
        }

        rt.sizeDelta = new Vector2(finalW, finalH);
    }
}