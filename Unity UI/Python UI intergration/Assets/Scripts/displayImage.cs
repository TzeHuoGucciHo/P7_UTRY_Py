using System.Diagnostics;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.IO;
using Mediapipe.Tasks.Vision.PoseLandmarker;
using Mediapipe.Tasks.Core;
using Mediapipe.Tasks.Vision.Core;
using SFB;
using Unity.VisualScripting; // Namespace for StandaloneFileBrowser


public class displayImage : MonoBehaviour
{
    public RawImage display;
    private PoseLandmarker poseLandmarker;
    private PoseLandmarkerOptions PoseLandmarkerOptions;
    private BaseOptions BaseOptions;
    
    
    private void Start()
    {
        
        var modelPath = Path.Combine(Application.streamingAssetsPath, "pose_landmarker_full.task");
        
        var options = new PoseLandmarkerOptions()
        {
            BaseOptions = new BaseOptions(modelAssetPath: modelPath),
            RunningMode = RunningMode.IMAGE
        };

        poseLandmarker = PoseLandmarker.CreateFromOptions(options);

        // Subscribe to output
        poseLandmarker.OnResult += OnPoseLandmarkResult;
    }

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
        Texture2D tex = new Texture2D(0,0);
        tex.LoadImage(fileData);
        display.texture = tex;
        Texture2D Uploadedtex = display.texture as Texture2D;
       
        

        yield return null;
    }
}