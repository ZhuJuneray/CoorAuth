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
    private float blinkCooldown = 0.5f;  // ���ñ��۳���ʱ����ֵ
    public OVRFaceExpressions ovrFaceExpressions;

    private OVRFaceExpressions.FaceExpression eyeCloseL = OVRFaceExpressions.FaceExpression.EyesClosedL;
    private OVRFaceExpressions.FaceExpression eyeCloseR = OVRFaceExpressions.FaceExpression.EyesClosedR;
    private float closeL;
    private float closeR;
    private float lastBlinkTime;
    private bool isUp = false; // ���
    private int pinNum = 4;

    private void Awake()
    {
        time = 1;
        //indexOfPIN = 0;
        //indexOfSize = 0;
        isUp = false;

        // ��ȡ��ǰ���ں�ʱ��
        DateTime currentDate = DateTime.Now;
        int month = currentDate.Month;
        int day = currentDate.Day;
        Name = "study1-" + Name + "-" + month.ToString() + day.ToString();
    }

    void Update()
    {
        
        // ��ȡ��ǰ֡�ı���״̬
        if (!ovrFaceExpressions.TryGetFaceExpressionWeight(eyeCloseL, out closeL)) return;
        if (!ovrFaceExpressions.TryGetFaceExpressionWeight(eyeCloseR, out closeR)) return;

        //Debug.Log("close" + closeL + "," + closeR);
        // ��������۶����ϣ�������һ�μ�⵽���۵�ʱ�䳬����ֵ�������Begin
        if (Mathf.Max(closeL, closeR) > 0.75)
        {
            // ��δ����
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
                // ���ڽ��У�ֹͣ������
                if (Begin)
                {
                    Begin = false;
                    isUp = false; // ��ԭisUp
                    return ;
                }
                // ��ʼ
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
                isUp = false; // ��ԭisUp
            }
            lastBlinkTime = Time.time;  // ��¼����δ���۵�ʱ��
        }
    }

    public bool getIsUp()
    {
        return isUp;
    }

}
