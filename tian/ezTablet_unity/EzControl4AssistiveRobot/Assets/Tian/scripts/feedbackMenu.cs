
using UnityEngine;

public class feedbackMenu : MonoBehaviour {
    public UIfunctions UIcontrol;
    public UnityEngine.UI.Text msgFeedback;
    public UnityEngine.UI.Button btnConfirm;
    public UnityEngine.UI.Button btnCancel;
    private void OnDisable()
    {
        UIcontrol.mode = 0;
        UIcontrol.confirmation = 0;
    }
    private void OnEnable()
    {
        msgFeedback.text = "Searching for solution...";
        btnCancel.gameObject.SetActive(false);
        btnConfirm.gameObject.SetActive(false);

    }
}
