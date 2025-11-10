using System.Diagnostics;
using UnityEngine;
using UnityEngine.UI;

public class displayNumbers : MonoBehaviour
{
    public InputField height;

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

        height.text = output.Trim();
    }
}
