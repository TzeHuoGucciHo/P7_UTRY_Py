using UnityEngine;
using TMPro;
using System;
using System.IO;
using UnityEngine.UI;

// This class name MUST match the type used in PythonMeasurementProcessor.cs
public class MeasurementDisplay : MonoBehaviour
{
    [Header("Main UI Output Fields (Initial View)")]
    // 1. For "Your size: S"
    public TMP_Text recommendedSizeText;
    // 2. For "Additional info: For a looser fit..." (This is the simple line)
    public TMP_Text simpleInfoText;

    [Header("Detailed Pop-up Output Field")]
    // 3. For the content of the white pop-up box (the detailed comparison lines)
    public TMP_Text detailedInfoPopupText;

    [Header("Image Output Field")]
    public RawImage textureDisplayArea;

    // --- Data Structures (Kept for consistency) ---
    [System.Serializable]
    public class MeasurementData
    {
        public float? height_cm;
        public float? chest_cm;
        public float? waist_cm;
        public float? hip_cm;
        public float? shoulder_width_cm;
    }
    // --------------------------------------------------------------------------

    /// <summary>
    /// Receives the final measurement data and updates the UI fields.
    /// This method now accepts 5 arguments, matching the call from PythonMeasurementProcessor.cs.
    /// </summary>
    /// <param name="finalMeasurementsJsonString">The JSON string containing the final imputed measurements.</param>
    /// <param name="simpleSizeString">The size only (e.g., "M").</param>
    /// <param name="imagePath">The full path to the 'front_overlay.png' image.</param>
    /// <param name="simpleFitDescription">The fit suffix and simple suggestion line (e.g., "(Can have bit loose fit)\nNext best size is M").</param>
    /// <param name="fullDetailedComparison">The comparison lines for the pop-up (e.g., "Compared with the next best size...").</param>
    public void DisplayMeasurementsFromPython(string finalMeasurementsJsonString, string simpleSizeString, string imagePath, string simpleFitDescription, string fullDetailedComparison)
    {

        // --- IMPORTANT CHECK (References) ---
        if (recommendedSizeText == null || simpleInfoText == null || detailedInfoPopupText == null)
        {
            Debug.LogError("Display Error: One or more required TMP_Text references are missing. Check assignments.");
            return;
        }

        // 1. Handle Recommended Size (Your size: S)
        // Ensure the output matches your desired format: "Your size: S"
        if (!string.IsNullOrEmpty(simpleSizeString))
        {
            recommendedSizeText.text = simpleSizeString;
        }
        else
        {
            recommendedSizeText.text = "Your size: N/A";
        }


        // 2. Handle Simple Info (Initial View) 
        // simpleFitDescription contains: "(Can have bit loose fit)\nNext best size is M"
        if (!string.IsNullOrEmpty(simpleFitDescription))
        {
            // We use this text directly for the simple info field
            // Note: The "Additional info:" prefix is added here to match your UI image.
            simpleInfoText.text = simpleFitDescription;
        }
        else
        {
            simpleInfoText.text = "Additional info: Best fit determined.";
        }

        // 3. Handle Detailed Comparison (Pop-up View)
        // fullDetailedComparison contains the full comparison lines (ready for the pop-up)
        if (!string.IsNullOrEmpty(fullDetailedComparison))
        {
            detailedInfoPopupText.text = fullDetailedComparison.Trim();
        }
        else
        {
            detailedInfoPopupText.text = "No detailed comparison available.";
        }


        // --- Load and display the Image ---
        if (textureDisplayArea != null)
        {
            Texture2D texture = LoadTextureFromFile(imagePath);
            if (texture != null)
            {
                textureDisplayArea.texture = texture;
                textureDisplayArea.color = Color.white;
                Debug.Log($"Successfully loaded and displayed image: {Path.GetFileName(imagePath)}");
            }
        }
        else
        {
            Debug.LogError("Texture Display Area (RawImage) is not assigned in the Inspector!");
        }
    }

    /// <summary>
    /// Loads a texture from a local file path using System.IO.
    /// </summary>
    private Texture2D LoadTextureFromFile(string filePath)
    {
        if (!File.Exists(filePath))
        {
            Debug.LogError($"File not found: {filePath}");
            return null;
        }

        try
        {
            byte[] fileData = File.ReadAllBytes(filePath);
            Texture2D texture = new Texture2D(2, 2);

            if (texture.LoadImage(fileData))
            {
                return texture;
            }
            else
            {
                Debug.LogError($"Could not load image data into Texture2D from file: {filePath}");
                return null;
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error loading texture from file {filePath}: {e.Message}");
            return null;
        }
    }
}