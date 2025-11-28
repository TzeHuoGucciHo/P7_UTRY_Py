using System;
using UnityEngine;
using UnityEngine.UI;
using SFB;
using System.IO;
using UnityEngine.EventSystems;
using TMPro;
using System.Collections;
using Debug = UnityEngine.Debug;

[RequireComponent(typeof(Button))]
public class UIscriptAndy : MonoBehaviour, IPointerClickHandler
{
    public TextMeshProUGUI Text;

    [HideInInspector]
    public string selectedFilePath = ""; // Stores the path to the selected original file

    public RawImage displayImage; // Drag your UI RawImage here in the inspector
    public GameObject Panel;
    public GameObject AdInfo;
    private void Start()
    {
        Panel.SetActive(false);
        AdInfo.SetActive(false);
        
    }

    public void OnPointerClick(PointerEventData eventData)
    {
        EventSystem.current.SetSelectedGameObject(null);
    }

    public void Hoverenter()
    {
        Text.color = Color.black;
    }

    public void Hoverexit()
    {
        Text.color = Color.white;
    }

    // --- IMAGE SELECTION ---
    public void OnClickChooseImage()
    {
        var extensions = new[] {
      new ExtensionFilter("Image Files", "png", "jpg", "jpeg")
    };
        var paths = StandaloneFileBrowser.OpenFilePanel("Select an Image", "", extensions, false);

        if (paths.Length > 0 && !string.IsNullOrEmpty(paths[0]))
        {
            // 1. Save the selected path (INPUT path)
         selectedFilePath = paths[0];

         // 2. Load and display the ORIGINAL image immediately
         StartCoroutine(LoadAndFitImage(paths[0]));
        }
    }


    // --- IMAGE LOADING AND FITTING ---
    private IEnumerator LoadAndFitImage(string filePath)
    {
        var fileData = File.ReadAllBytes(filePath);
        Texture2D tex = new Texture2D(2, 2);
        tex.LoadImage(fileData);

        displayImage.texture = tex;

        // Use the fitting logic to size the image
        FitRawImageToParent(displayImage, tex);

        yield return null;
    }

    void FitRawImageToParent(RawImage raw, Texture2D tex)
    {
        RectTransform rt = raw.rectTransform;
        RectTransform parent = raw.transform.parent.GetComponent<RectTransform>();

        if (parent == null) return;

        float texW = tex.width;
        float texH = tex.height;

        float parentW = parent.rect.width;
        float parentH = parent.rect.height;

        float texAspect = texW / texH;
        float parentAspect = parentW / parentH;

        float finalW, finalH;

        if (texAspect > parentAspect)
        {
             // Fit by width
            finalW = parentW;
            finalH = finalW / texAspect;
        }
        else
        {
             // Fit by height
            finalH = parentH;
            finalW = finalH * texAspect;
        }

        rt.sizeDelta = new Vector2(finalW, finalH);
    }
}