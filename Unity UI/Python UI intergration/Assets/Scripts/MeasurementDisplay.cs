using UnityEngine;
using TMPro;
using System;

// This class name MUST match the type used in PythonMeasurementProcessor.cs
public class MeasurementDisplay : MonoBehaviour
{
    [Header("UI Output Fields (Assign in Inspector)")]
    // Measurement Input Fields (Kept as InputField for consistency with your measurement display)
    public TMP_InputField heightInput;
    public TMP_InputField chestWidthInput;
    public TMP_InputField bodyLengthInput;
    public TMP_InputField sleeveLengthInput;

    [Header("Size Output Fields (Assign in Inspector)")]
    // FIX: Changed from TMP_InputField to TMP_Text
    public TMP_Text recommendedSizeText; // Now a Text component
    public TMP_Text additionalInfoText; // Now a Text component


    // --- Data structure to match the JSON keys from Python's final_measurements_json ---
    [Serializable]
    public class MeasurementsJsonStructure
    {
        // CRITICAL FIX: These fields now match the EXACT keys from your Python JSON output.
        public float TotalHeight;       // Maps to height_cm
        public float ChestFrontWidth;   // Maps to chest_width_cm
        public float ShoulderToWaist;   // Maps to body_length_cm (closest metric)
        public float ArmLength;         // Maps to sleeve_length_cm (closest metric)

        // Include other keys from the JSON to avoid errors, even if unused
        public float Gender;
        public float Age;
        public float HeadCircumference;
        public float ShoulderWidth;
        public float ChestCircumference;
        public float Belly;
        public float Waist;
        public float Hips;
        public float WaistToKnee;
        public float LegLength;
    }

    /// <summary>
    /// Receives the final measurement JSON string and updates the UI input fields.
    /// This method is called by PythonMeasurementProcessor.cs.
    /// </summary>
    /// <param name="finalMeasurementsJsonString">The JSON string containing the final imputed measurements.</param>
    /// <param name="recommendedSizeString">The raw string containing the size and comparison info.</param>
    public void DisplayMeasurementsFromPython(string finalMeasurementsJsonString, string recommendedSizeString)
    {
        // --- IMPORTANT CHECK (Measurements) ---
        if (heightInput == null || chestWidthInput == null || bodyLengthInput == null || sleeveLengthInput == null)
        {
            Debug.LogError("Display Error: One or more TMP_InputField references are missing. Check assignments.");
            return;
        }

        // --- IMPORTANT CHECK (Size Info) ---
        // Check for TMP_Text references
        if (recommendedSizeText == null || additionalInfoText == null)
        {
            Debug.LogError("Display Error: One or more TMP_Text references for size output are missing. Check assignments.");
            // We can continue with measurement display if these are the only issues
        }

        if (string.IsNullOrEmpty(finalMeasurementsJsonString))
        {
            Debug.LogError("Display Error: Received an empty or null JSON string for measurements.");
        }
        else
        {
            try
            {
                // 1. Deserialize the JSON string into the C# structure
                MeasurementsJsonStructure data = JsonUtility.FromJson<MeasurementsJsonStructure>(finalMeasurementsJsonString);

                // 2. Update the UI fields, using the values retrieved via the Python key names.
                // Measurement fields (still InputField)
                heightInput.text = $"{data.TotalHeight:F1}";
                chestWidthInput.text = $"{data.ChestFrontWidth:F1}";
                bodyLengthInput.text = $"{data.ShoulderToWaist:F1}";
                sleeveLengthInput.text = $"{data.ArmLength:F1}";

                Debug.Log("Successfully parsed JSON and updated all UI measurement fields.");
            }
            catch (Exception e)
            {
                Debug.LogError($"Display Error: Failed to parse or use JSON data. Exception: {e.Message}");
                Debug.Log($"Raw JSON received: {finalMeasurementsJsonString}");
            }
        }

        // --- Handle Recommended Size and Additional Info ---
        if (!string.IsNullOrEmpty(recommendedSizeString))
        {
            // The format is: "Recommended: XXL\n(Compared to the next best size (XL): ...)"
            string[] parts = recommendedSizeString.Split(new[] { '\n' }, 2);

            if (recommendedSizeText != null)
            {
                recommendedSizeText.text = parts[0]; // e.g., "Recommended: XXL"
            }

            if (additionalInfoText != null && parts.Length > 1)
            {
                // Display the rest of the text
                string info = parts[1].Trim();
                // Ensure it's wrapped for consistency, if not already
                additionalInfoText.text = info.StartsWith("(") ? info : $"({info})";
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
    }
}