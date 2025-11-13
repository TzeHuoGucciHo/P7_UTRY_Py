using System;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;

public class WebFunctions : MonoBehaviour
{
    public Sprite img1;
    public Sprite img2;
    public Sprite img3;
    public Sprite img4;
    public Sprite img5;

    public Image displayImg;

    public GameObject popup;

    private void Start()
    {
        popup.SetActive(false);
    }

    public void ImageLoad()
    {
        SceneManager.LoadScene("UtryMain");
    }

    public void ManuelLoad()
    {
        SceneManager.LoadScene("Manuel input");
    }

    public void changeImg1()
    {
        displayImg.sprite = img1;
    }
    public void changeImg2()
    {
        displayImg.sprite = img2;
    }
    public void changeImg3()
    {
        displayImg.sprite = img3;
    }
    public void changeImg4()
    {
        displayImg.sprite = img4;
    }
    public void changeImg5()
    {
        displayImg.sprite = img5;
    }
    
}
