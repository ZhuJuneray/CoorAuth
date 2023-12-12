using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class FaceExpressionsTracking : MonoBehaviour
{
    [SerializeField]
    public OVRFaceExpressions ovrFaceExpressions;
    private AuthController trackingScript;
    private bool Begin;
    private string Name;
    public List<FeatureTracking> featureList = new List<FeatureTracking>();
    private string pathStr;
    private Dictionary<OVRFaceExpressions.FaceExpression, List<float>> keyValueExpressionsPairs = new();
    private void Awake()
    {
        trackingScript = GameObject.Find("Controller").GetComponent<AuthController>();
    }
    // Start is called before the first frame update
    void Start()
    {
        if (ovrFaceExpressions == null)
            Debug.LogError("no tracking target");
        else
            Debug.Log("Tracking target has been found");
        trackingScript = GameObject.Find("Controller").GetComponent<AuthController>();
        if (trackingScript == null)
        {
            Debug.LogError("Can't find controller");
        }
        Begin = trackingScript.Begin;
        Name = trackingScript.Name;
    }

    // Update is called once per frame
    void Update()
    {
        Begin = trackingScript.Begin;
    }

    public void FixedUpdate()
    {
        if (Begin)
        {
            //ȡ����ƽ��ֵ
            float wTmp = 0;
            //Debug.Log(featureList.Count);
            foreach (FeatureTracking featureTracking in featureList)
            {
                float weight = featureTracking.isTracking ? ovrFaceExpressions.GetWeight(featureTracking.expressions) : -1;
                if (featureTracking.expressions.ToString().EndsWith("L")) wTmp = weight;
                else
                {
                    //Debug.Log(featureTracking.expressions.ToString() + weight + " " + wTmp);
                    if (keyValueExpressionsPairs.ContainsKey(featureTracking.expressions))
                        keyValueExpressionsPairs[featureTracking.expressions].Add((wTmp + weight) / 2);
                    else
                        keyValueExpressionsPairs.Add(featureTracking.expressions, new List<float>());
                }
            }
        }
    }
    public void OnDestroy()
    {
        pathStr = @"E:\Desktop\data\VRAuth\Expression_data_" + Name + ".csv";
        //Debug.Log(keyValueExpressionsPairs.Count + "length" + keyValueExpressionsPairs[OVRFaceExpressions.FaceExpression.LidTightenerR].Count);
        using (StreamWriter writer = new StreamWriter(pathStr))
        {

            writer.WriteLine("FacialBlendshapes");
            // �����ֵ䲢д��ÿһ��
            foreach (var kvp in keyValueExpressionsPairs)
            {
                var expression = kvp.Key;
                var values = kvp.Value;
                // �����ݸ�ʽ��Ϊ CSV ��
                string csvLine = $"{expression},{string.Join(",", values)}";
                writer.WriteLine(csvLine);
            }
            Debug.Log("Finish Writing FaceFeatures");

        }
    }
}