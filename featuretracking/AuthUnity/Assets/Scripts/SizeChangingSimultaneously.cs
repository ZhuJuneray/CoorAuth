using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SizeChangingSimultaneously : MonoBehaviour
{
    public float scaleFactor = 1.5f; // 变大的倍数
    public float scaleSpeed = 0.5f; // 变换速度
    public float minScale = 0.5f; // 最小缩放比例
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
            // 根据变大或变小的状态来调整缩放因子
            float scaleChange = isScalingUp ? scaleSpeed * Time.deltaTime : -scaleSpeed * Time.deltaTime;
            transform.localScale += new Vector3(scaleChange, scaleChange, scaleChange);
            if (transform.localScale.x >= scaleFactor || transform.localScale.x <= minScale)
            {
                isScalingUp = !isScalingUp;
            }
        }
    }
}
