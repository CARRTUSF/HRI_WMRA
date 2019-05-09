using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TimeBomb : MonoBehaviour {

	void OnEnable ()
    {
        StartCoroutine(Timebomb());
	}

    IEnumerator Timebomb()
    {
        yield return new WaitForSeconds(3);
        gameObject.SetActive(false);        
    }
}
