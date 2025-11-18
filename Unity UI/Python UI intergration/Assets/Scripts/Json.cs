using UnityEngine;
using TMPro;
using System.IO;

[System.Serializable]
public class BodyMeasurements
{
    public float chest_cm;
    public float waist_cm;
    public float hip_cm;
    public float thigh_cm;
    public float shoulder_width_cm;
}

public class Json : MonoBehaviour
{
    public string jsonFilePath; // Full path to your JSON
    public TMP_InputField chestInput;
    public TMP_InputField waistInput;
    public TMP_InputField hipInput;
    public TMP_InputField thighInput;
    public TMP_InputField shoulderInput;

    void Start()
    {
        if (!File.Exists(jsonFilePath))
        {
            Debug.LogError($"JSON file not found at path: {jsonFilePath}");
            return;
        }

        string jsonText = File.ReadAllText(jsonFilePath);

        // Wrap JSON in an object with the fields you care about
        BodyMeasurements data = JsonUtility.FromJson<BodyMeasurements>(jsonText);

        chestInput.text = data.chest_cm.ToString("F2");
        waistInput.text = data.waist_cm.ToString("F2");
        hipInput.text = data.hip_cm.ToString("F2");
        thighInput.text = data.thigh_cm.ToString("F2");
        shoulderInput.text = data.shoulder_width_cm.ToString("F2");
    }
}