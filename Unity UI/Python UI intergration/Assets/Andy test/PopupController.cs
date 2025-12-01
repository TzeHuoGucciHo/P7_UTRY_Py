using UnityEngine;
using UnityEngine.UI;

public class PopupController : MonoBehaviour
{
    public Button infoButton;        // Assign in inspector
    public Button closeButton;       // Assign in inspector
    public GameObject infoPopup;     // Assign in inspector

    private bool isActivated = false;

    void Start()
    {
        infoPopup.SetActive(false);

        infoButton.onClick.AddListener(OnInfoClick);
        closeButton.onClick.AddListener(OnCloseClick);
    }

    void OnInfoClick()
    {
        isActivated = true;
        infoPopup.SetActive(true);
    }

    void OnCloseClick()
    {
        isActivated = false;
        infoPopup.SetActive(false);
    }
}
