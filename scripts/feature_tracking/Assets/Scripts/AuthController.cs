using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class AuthController : MonoBehaviour
{
    [SerializeField]
    public bool Begin;
    public string Name;
    public int indexOfSize;
    public int indexOfPIN;
    public int time;
    [SerializeField]
    [Range(0f, 1f)]
    private float blinkCooldown = 0.5f;  // 设置闭眼持续时间阈值
    public OVRFaceExpressions ovrFaceExpressions;

    private OVRFaceExpressions.FaceExpression eyeCloseL = OVRFaceExpressions.FaceExpression.EyesClosedL;
    private OVRFaceExpressions.FaceExpression eyeCloseR = OVRFaceExpressions.FaceExpression.EyesClosedR;
    private float closeL;
    private float closeR;
    private float lastBlinkTime;
    private bool isUp = false; // 标记
    private int pinNum = 4;

    private void Awake()
    {
        time = 1;
        //indexOfPIN = 0;
        //indexOfSize = 0;
        isUp = false;

        // 获取当前日期和时间
        DateTime currentDate = DateTime.Now;
        int month = currentDate.Month;
        int day = currentDate.Day;
        Name = "study1-" + Name + "-" + month.ToString() + day.ToString();
    }

    void Update()
    {
        
        // 获取当前帧的闭眼状态
        if (!ovrFaceExpressions.TryGetFaceExpressionWeight(eyeCloseL, out closeL)) return;
        if (!ovrFaceExpressions.TryGetFaceExpressionWeight(eyeCloseR, out closeR)) return;

        //Debug.Log("close" + closeL + "," + closeR);
        // 如果左右眼都闭上，并且上一次检测到闭眼的时间超过阈值，则更新Begin
        if (Mathf.Max(closeL, closeR) > 0.75)
        {
            // 尚未更新
            if (Time.time - lastBlinkTime > blinkCooldown)
            {
                isUp = true;
                //Debug.Log("Eyes closed for more than n seconds. Begin updated");
            }
        }
        else
        {
            if (isUp == true)
            {
                // 正在进行，停止并返回
                if (Begin)
                {
                    Begin = false;
                    isUp = false; // 还原isUp
                    return ;
                }
                // 开始
                else Begin = true;

                if (indexOfPIN == 0)
                {
                    indexOfPIN = indexOfPIN + 1;
                    indexOfSize = indexOfSize + 1;
                }
                else if (indexOfPIN == pinNum)
                {

                }
                else indexOfPIN += 1;
                isUp = false; // 还原isUp
            }
            lastBlinkTime = Time.time;  // 记录本次未闭眼的时间
        }
    }

    public bool getIsUp()
    {
        return isUp;
    }

}
