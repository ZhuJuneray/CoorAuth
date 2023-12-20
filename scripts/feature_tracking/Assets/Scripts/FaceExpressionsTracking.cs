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
            //取左右平均值
            float wTmp = 0;
            //Debug.Log(featureList.Count);
            foreach (FeatureTracking featureTracking in featureList)
            {
                float weight = -1;
                if (featureTracking.isTracking)
                    ovrFaceExpressions.TryGetFaceExpressionWeight(featureTracking.expressions, out weight);
                if (featureTracking.expressions.ToString().EndsWith("L")) wTmp = weight;
                else if(keyValueExpressionsPairs.ContainsKey(featureTracking.expressions) && featureTracking.expressions == OVRFaceExpressions.FaceExpression.JawDrop)
                    keyValueExpressionsPairs[featureTracking.expressions].Add(weight);
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
            List<string> l = new();
            for (int i = 0; i < 1000; i++) l.Add(i.ToString());
            writer.WriteLine(string.Join(", ", l));
            // 遍历字典并写入每一行
            foreach (var kvp in keyValueExpressionsPairs)
            {
                var expression = kvp.Key;
                var values = kvp.Value;
                // 将数据格式化为 CSV 行
                string csvLine = $"{expression},{string.Join(",", values)}";
                writer.WriteLine(csvLine);
            }
            Debug.Log("Finish Writing FaceFeatures");

        }
    }
}