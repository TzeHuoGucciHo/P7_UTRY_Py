using System;
using UnityEngine;
using UnityEngine.UI;
using SFB; // Namespace for StandaloneFileBrowser
using System.IO;
using NUnit.Framework;
using UnityEngine.EventSystems;
using TMPro;

public class UIscriptAndy : MonoBehaviour, IPointerClickHandler, IPointerEnterHandler, IPointerExitHandler
{
    public TextMeshProUGUI Text;
    public Color Utry = new Color(234, 88, 12);

    // This public variable is correctly accessible by the other script.
    [HideInInspector]
    public string selectedFilePath = "";

    public RawImage displayImage; // Drag your UI RawImage here in the inspector

    public void OnPointerEnter(PointerEventData eventdata)
    {
        Text.color = Utry;
    }

    public void OnPointerExit(PointerEventData eventdata)
    {
        Text.color = Color.white;
    }

    public void OnPointerClick(PointerEventData eventData)
    {
        EventSystem.current.SetSelectedGameObject(null);
    }

    public void OnClickChooseImage()
    {
        var extensions = new[] {
            new ExtensionFilter("Image Files", "png", "jpg", "jpeg")
        };
        var paths = StandaloneFileBrowser.OpenFilePanel("Select an Image", "", extensions, false);

        if (paths.Length > 0 && !string.IsNullOrEmpty(paths[0]))
        {
            selectedFilePath = paths[0];
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