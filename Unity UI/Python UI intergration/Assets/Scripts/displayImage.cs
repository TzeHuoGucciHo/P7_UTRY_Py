using System.Diagnostics;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.IO;
using SFB; // Namespace for StandaloneFileBrowser

public class displayImage : MonoBehaviour
{
    public RawImage display; // Drag your UI RawImage here in the inspector

    public void OnClickChooseImage()
    {
        // Open a file browser window
        var extensions = new[] {
            new ExtensionFilter("Image Files", "png", "jpg", "jpeg")
        };
        var paths = StandaloneFileBrowser.OpenFilePanel("Select an Image", "", extensions, false);

        if (paths.Length > 0 && !string.IsNullOrEmpty(paths[0]))
        {
            StartCoroutine(LoadImage(paths[0]));
        }
    }

    private System.Collections.IEnumerator LoadImage(string filePath)
    {
        var fileData = File.ReadAllBytes(filePath);
        Texture2D tex = new Texture2D(2, 2);
        tex.LoadImage(fileData);
        display.texture = tex;
        
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
        RectTransform rt = display.rectTransform;
        rt.sizeDelta = new Vector2(newW, newH);
        
        yield return null;
    }
}