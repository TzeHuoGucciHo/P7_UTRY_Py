using System.Diagnostics;
using UnityEngine;
using UnityEngine.UI;
using System;
using System.IO;

public class PythonRunner : MonoBehaviour
{
    public RawImage imageDisplay;          // Assign in inspector
    public string pythonExePath;           // Path to python.exe
    public string scriptPath;              // The python script
    public string outputImagePath = "cropped_output.png";

    public void RunPythonScript()
    {
        ProcessStartInfo psi = new ProcessStartInfo();
        psi.FileName = pythonExePath;          // example: C:/Python312/python.exe
        psi.Arguments = "\"" + scriptPath + "\"";
        psi.CreateNoWindow = true;
        psi.UseShellExecute = false;
        psi.RedirectStandardOutput = true;
        psi.RedirectStandardError = true;

        Process process = new Process();
        process.StartInfo = psi;

        process.Start();
        process.WaitForExit();  // wait until python is done
        
        

        LoadCroppedImage();
    }

    void LoadCroppedImage()
    {
        if (!File.Exists(outputImagePath))
        {
            
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

        float texW = tex.width;
        float texH = tex.height;

        float parentW = parent.rect.width;
        float parentH = parent.rect.height;

        float texAspect = texW / texH;
        float parentAspect = parentW / parentH;

        float finalW, finalH;

        if (texAspect > parentAspect)
        {
            // Image is wider relative to parent → width limits size
            finalW = parentW;
            finalH = finalW / texAspect;
        }
        else
        {
            // Image is taller relative to parent → height limits size
            finalH = parentH;
            finalW = finalH * texAspect;
        }

        rt.sizeDelta = new Vector2(finalW, finalH);
    }
}
