using System;
using UnityEngine;
using UnityEngine.UI;
using SFB; // Namespace for StandaloneFileBrowser
using System.IO;
using NUnit.Framework;
using UnityEngine.EventSystems;
using TMPro;
using Unity.VisualScripting.Antlr3.Runtime.Tree;
using UnityEngine.SceneManagement;

public class UIscript : MonoBehaviour
{
    public Button[] buttons; 
    public Color UtryBlue = new Color(2, 2, 2);
    public Color UtryOrange = new Color(2,2,2);

    public Image explain;

    public void ChangeButtonColor(Button btn)
    {
        btn.GetComponentInChildren<TMP_Text>().color = UtryOrange;
        btn.GetComponent<Image>().color = Color.white;

    }

    public void PointerExit(Button btn)
    {
        btn.GetComponentInChildren<TMP_Text>().color = Color.white;
        btn.GetComponent<Image>().color = UtryOrange;
    }

    public void ConfirmSizes()
    {
        explain.enabled = false;
        
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
            StartCoroutine(LoadImage(paths[0]));
        }
    }

    private System.Collections.IEnumerator LoadImage(string filePath)
    {
        var fileData = File.ReadAllBytes(filePath);
        Texture2D tex = new Texture2D(2, 2);
        tex.LoadImage(fileData);
        displayImage.texture = tex;
        RectTransform rt = displayImage.rectTransform;
        rt.sizeDelta = new Vector2(420, 890);
        yield return null;
    }

    
}