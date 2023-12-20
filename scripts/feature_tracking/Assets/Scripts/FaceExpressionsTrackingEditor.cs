using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

[System.Serializable]
public class FeatureTracking
{
    public OVRFaceExpressions.FaceExpression expressions;
    public bool isTracking;
    public FeatureTracking() { }
    public FeatureTracking(OVRFaceExpressions.FaceExpression f)
    {
        expressions = f;
        isTracking = true;
    }
}

[CustomEditor(typeof(FaceExpressionsTracking))] // Ϊ��ָ���Զ���༭��
public class FaceExpressionsTrackingEditor : Editor
{
    public override void OnInspectorGUI()
    {
        FaceExpressionsTracking script = (FaceExpressionsTracking)target;

        // ��ʾĬ��Inspector����
        DrawDefaultInspector();

        EditorGUILayout.Space();

        // ���һ����ť���Ա���Inspector�г�ʼ��featureList
        if (GUILayout.Button("Initialize featureList"))
        {
            InitializeFeatureList(script);
        }
    }

    private void InitializeFeatureList(FaceExpressionsTracking script)
    {
        script.featureList.Clear();

        foreach (FeatureTracking features in interstedFeatures)
        {
            script.featureList.Add(features);
        }
        
    }

    private readonly FeatureTracking[] interstedFeatures = new FeatureTracking[]
    {
        new FeatureTracking(OVRFaceExpressions.FaceExpression.NoseWrinklerL),
        new FeatureTracking(OVRFaceExpressions.FaceExpression.NoseWrinklerR),
        new FeatureTracking(OVRFaceExpressions.FaceExpression.CheekRaiserL),
        new FeatureTracking(OVRFaceExpressions.FaceExpression.CheekRaiserR),
        new FeatureTracking(OVRFaceExpressions.FaceExpression.LidTightenerL),
        new FeatureTracking(OVRFaceExpressions.FaceExpression.LidTightenerR),
        new FeatureTracking(OVRFaceExpressions.FaceExpression.UpperLipRaiserL),
        new FeatureTracking(OVRFaceExpressions.FaceExpression.UpperLipRaiserR),
        new FeatureTracking(OVRFaceExpressions.FaceExpression.EyesClosedL),
        new FeatureTracking(OVRFaceExpressions.FaceExpression.EyesClosedR),
        new FeatureTracking(OVRFaceExpressions.FaceExpression.UpperLidRaiserL),
        new FeatureTracking(OVRFaceExpressions.FaceExpression.UpperLidRaiserR),
        new FeatureTracking(OVRFaceExpressions.FaceExpression.EyesLookUpL),
        new FeatureTracking(OVRFaceExpressions.FaceExpression.EyesLookUpR),
        new FeatureTracking(OVRFaceExpressions.FaceExpression.BrowLowererL),
        new FeatureTracking(OVRFaceExpressions.FaceExpression.BrowLowererR),
        new FeatureTracking(OVRFaceExpressions.FaceExpression.OuterBrowRaiserL),
        new FeatureTracking(OVRFaceExpressions.FaceExpression.OuterBrowRaiserR),
        new FeatureTracking(OVRFaceExpressions.FaceExpression.InnerBrowRaiserL),
        new FeatureTracking(OVRFaceExpressions.FaceExpression.InnerBrowRaiserR),
        //new FeatureTracking(OVRFaceExpressions.FaceExpression.JawDrop),
    };
}