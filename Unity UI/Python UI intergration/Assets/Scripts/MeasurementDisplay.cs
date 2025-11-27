using UnityEngine;
using TMPro;
using System;
using System.IO;       // Needed for File and Path operations
using UnityEngine.UI;    // Needed for RawImage

// This class name MUST match the type used in PythonMeasurementProcessor.cs
public class MeasurementDisplay : MonoBehaviour
{
    [Header("Size Output Fields (Assign in Inspector)")]
    // FIX: Changed from TMP_InputField to TMP_Text
    public TMP_Text recommendedSizeText; // Now a Text component
    public TMP_Text additionalInfoText; // Now a Text component

    [Header("Image Output Field")]
    // Assign a RawImage component here in the Inspector!
    public RawImage textureDisplayArea;

    // --- Data Structures (Kept for consistency, you can remove if unused) ---
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
    /// Receives the final measurement JSON string and updates the UI input fields.
    /// This method is called by PythonMeasurementProcessor.cs.
    /// NOTE: The signature must be updated in PythonMeasurementProcessor to include 'imagePath'.
    /// </summary>
    /// <param name="finalMeasurementsJsonString">The JSON string containing the final imputed measurements.</param>
    /// <param name="recommendedSizeString">The raw string containing the size and comparison info.</param>
    /// <param name="imagePath">The full path to the 'front_overlay.png' image.</param>
    public void DisplayMeasurementsFromPython(string finalMeasurementsJsonString, string recommendedSizeString, string imagePath)
    {

        // --- IMPORTANT CHECK (Size Info) ---
        if (recommendedSizeText == null || additionalInfoText == null)
        {
            Debug.LogError("Display Error: One or more TMP_Text references for size output are missing. Check assignments.");
        }

        // --- Handle Recommended Size and Additional Info ---
        if (!string.IsNullOrEmpty(recommendedSizeString))
        {
            // The format is: "Recommended: XXL\n(Compared to the next best size (XL): ...)"
            string[] parts = recommendedSizeString.Split(new[] { '\n' }, 2);

            if (recommendedSizeText != null)
            {
                // e.g., "Recommended: XXL"
                recommendedSizeText.text = parts[0];
            }

            if (additionalInfoText != null && parts.Length > 1)
            {
                // Display the rest of the text
                string info = parts[1].Trim();

                // --- FIX APPLIED HERE ---
                // REMOVE the logic that wraps the string in parentheses.
                additionalInfoText.text = info;
            }
            else if (additionalInfoText != null)
            {
                // If there's no second part, display a default message
                additionalInfoText.text = "No detailed comparison available.";
            }
        }
        else if (recommendedSizeText != null)
        {
            recommendedSizeText.text = "Size N/A";
            if (additionalInfoText != null) additionalInfoText.text = "";
        }

        // --- NEW: Load and display the Image ---
        if (textureDisplayArea != null)
        {
            Texture2D texture = LoadTextureFromFile(imagePath);
            if (texture != null)
            {
                // Assign the loaded texture to the RawImage component
                textureDisplayArea.texture = texture;
                textureDisplayArea.color = Color.white; // Ensure it's not tinted
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
            // Texture2D size is a placeholder; LoadImage will resize it.
            Texture2D texture = new Texture2D(2, 2);

            // Load image data into the texture (reads dimensions/format from the file)
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