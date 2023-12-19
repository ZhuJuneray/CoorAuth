using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SizeChangingSimultaneously : MonoBehaviour
{
    public float scaleFactor = 1.5f; // ���ı���
    public float scaleSpeed = 0.5f; // �任�ٶ�
    public float minScale = 0.5f; // ��С���ű���
    private AuthController trackingScript;
    private bool isScalingUp = true;

    public void Start()
    {
        trackingScript = GameObject.Find("Controller").GetComponent<AuthController>();
    }

    void Update()
    {
        if (trackingScript.Begin)
        {
            // ���ݱ����С��״̬��������������
            float scaleChange = isScalingUp ? scaleSpeed * Time.deltaTime : -scaleSpeed * Time.deltaTime;
            transform.localScale += new Vector3(scaleChange, scaleChange, scaleChange);
            if (transform.localScale.x >= scaleFactor || transform.localScale.x <= minScale)
            {
                isScalingUp = !isScalingUp;
            }
        }
    }
}
