using System;
using UnityEngine;
using UnityEngine.UI;
using SFB; // Namespace for StandaloneFileBrowser
using System.IO;
using System.Windows.Forms;
using TMPro;
using UnityEngine.Rendering;
using Button = UnityEngine.UI.Button;
using System.Diagnostics;

public class UIscript : MonoBehaviour
{

    public string selectedFilePath = "";


    public GameObject explain;

    private void Start()
    {
        explain.SetActive(false);
    }

    public void ConfirmSizes()
    {
        explain.SetActive(true);
    }
      
    
    public RawImage displayImage; // Drag your UI RawImage here in the inspector

    public void OnClickChooseImage()
    {
        // Open a file browser window
        var extensions = new[] {
            new ExtensionFilter("Image Files", "png", "jpg", "jpeg")
        };
        var paths = StandaloneFileBrowser.OpenFilePanel("Select an Image", "", extensions, false);

        if (paths.Length > 0 && !string.IsNullOrEmpty(paths[0]))
        {
            selectedFilePath = paths[0];
            StartCoroutine(LoadImage(paths[0]));
        }
    }

    private System.Collections.IEnumerator LoadImage(string filePath)
    {
        var fileData = File.ReadAllBytes(filePath);
        Texture2D tex = new Texture2D(2, 2);
        tex.LoadImage(fileData);
        displayImage.texture = tex;
        
        // Max allowed size for the RawImage
        float maxW = 400f;
        float maxH = 900f;

        float imgW = tex.width;
        float imgH = tex.height;

        // Scale factor to fit within max area (keeps aspect ratio)
        float scale = Mathf.Min(maxW / imgW, maxH / imgH);

        // New size
        float newW = imgW * scale;
        float newH = imgH * scale;
        RectTransform rt = displayImage.rectTransform;
        rt.sizeDelta = new Vector2(newW, newH);
        
        yield return null;
    }
    
    public void RunPythonCrop(string imagePath)
    {
        string pythonExe = @"C:\Path\To\python.exe"; 
        string scriptPath = @"C:\Path\To\crop_script.py";

        ProcessStartInfo psi = new ProcessStartInfo();
        psi.FileName = pythonExe;
        psi.Arguments = $"\"{scriptPath}\" \"{imagePath}\"";
        psi.CreateNoWindow = true;
        psi.UseShellExecute = false;
        psi.RedirectStandardOutput = true;
        psi.RedirectStandardError = true;

        Process p = Process.Start(psi);

        string output = p.StandardOutput.ReadToEnd();
        string error = p.StandardError.ReadToEnd();

        p.WaitForExit();

        UnityEngine.Debug.Log("PYTHON OUTPUT:\n" + output);
        if (!string.IsNullOrEmpty(error))
            UnityEngine.Debug.LogError("PYTHON ERROR:\n" + error);
    }
    
}