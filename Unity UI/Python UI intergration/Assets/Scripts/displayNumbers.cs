using System.Diagnostics;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class displayNumbers : MonoBehaviour
{
    public TMP_InputField height;
    public TMP_InputField shoulderWidth;
    public TMP_InputField waistWidth;
    public TMP_InputField hipWidth;

    public void getNumbers()
    {
        //settings for unity to read python
        ProcessStartInfo psi = new ProcessStartInfo();
        psi.FileName = "pathtopython";
        psi.Arguments = "pathtoscript";
        psi.UseShellExecute = false;
        psi.RedirectStandardOutput = true;
        psi.CreateNoWindow = true;

        //runs the python script 
        Process p = Process.Start(psi);
        string output = p.StandardOutput.ReadToEnd();
        p.WaitForExit();
        string[] pairs = output.Split(',');

        height.text = pairs[0];
        shoulderWidth.text = pairs[1];
        waistWidth.text = pairs[2];
        hipWidth.text = pairs[3];

    }
}