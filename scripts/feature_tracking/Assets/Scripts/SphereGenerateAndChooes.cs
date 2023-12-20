using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class SphereGenerateAndChooes : MonoBehaviour
{
    [SerializeField]
    private float radius = 0.1f;
    [SerializeField]
    private float gazeDurationThreshold = 0.1f; // ����Ŀ��ͣ������ֵʱ��
    private float gazeTimer = 0f; // ���ڼ�ʱĿ��ͣ����ʱ��
    private float highHead = 1.35f;

    private AuthController trackingScript;
    private GazeVisualization gazeVisualization;
    private bool Begin, oldBegin;
    private bool isUp; // ��־
    //float[] angles = new float[] {0, 10, 20, 30, 40 };
    float[] angles = new float[] { 0, 8, 16, 24, 32 };
    private GameObject[,] spheres; // ���ڴ洢���ɵ��������
    private List<GameObject> selectedSpheres = new();// �Ѿ�ѡ���С������
    private List<GameObject> lineObjects = new(); // ���ɵĹ켣�Ķ���
    private List<LineRenderer> drawnLines = new List<LineRenderer>();
    List<int> order = new(); // �����PIN��order

    private Transform t;
    private float distance;
    // 3*3 ��С �Լ� ��Ӧ��PIN˳��
    private int index = 1;
    List<float> dis = new List<float>();
    private int numRows = 3; // 3 rows for a 3x3 grid
    private int numCols = 3; // 3 columns for a 3x3 grid
    private int cpyRow, cpyCol;
    // Start is called before the first frame update
    private void Awake()
    {
        t = GameObject.Find("GazeCalibration").transform;
        trackingScript = GameObject.Find("Controller").GetComponent<AuthController>();
        gazeVisualization = GameObject.Find("Controller").GetComponent<GazeVisualization>();

    }
    void Start()
    {
        if (trackingScript)
        {
            Begin = trackingScript.Begin;
            index = trackingScript.indexOfSize;
        }

        foreach (var a in angles)
        {
            dis.Add(Mathf.Tan(Mathf.Deg2Rad * a) * t.position.z);
            //Debug.Log(Mathf.Tan(Mathf.Deg2Rad * a) * t.position.z);
        }

        // ��ʼ�� spheres ����
        spheres = new GameObject[numRows, numCols];
        // 3*3���ĵ�
        Vector3 center = new Vector3(0, highHead, t.position.z);
        Debug.Log("index" + index);
        // ��һ�γ�ʼ��(��ʱindexΪ0)
        distance = dis[index + 1];
        GenerateSphere(distance, center);
    }

    // Update is called once per frame
    void Update()
    {
        // ��ʼ
        Begin = trackingScript.Begin;
        index = trackingScript.indexOfSize;
        if (Begin != oldBegin) ResetSpheresColor();
        if (!Begin) // δ��ʼ
        {
            return;
        }
        // �õ�ע�ӵ�
        Vector3 gazePoint = gazeVisualization.GetGazePos();
        int selectedRow = Mathf.Clamp(Mathf.FloorToInt((gazePoint.y - highHead + numRows * distance / 2) / distance), 0, numRows - 1);
        int selectedCol = Mathf.Clamp(Mathf.FloorToInt((gazePoint.x - t.position.x + numCols * distance / 2) / distance), 0, numCols - 1);
        //Debug.Log("row" + selectedRow + "cal" + selectedCol + "dis" + distance);
        HandleSelection(selectedRow, selectedCol);
        // ����״̬
        cpyCol = selectedCol;
        cpyRow = selectedRow;
    }

    private void LateUpdate()
    {
        isUp = trackingScript.getIsUp();
        oldBegin = Begin;
    }

    public void GenerateSphere(float distance, Vector3 center)
    {
        List<Vector3[,]> gridPoints = new List<Vector3[,]>();
        float deltaX = distance;
        float deltaY = distance;
        Debug.Log("dis" + distance);
        Vector3[,] currentGrid = new Vector3[numRows, numCols];

        // Generate grid points
        for (int row = 0; row < numRows; row++)
        {
            for (int col = 0; col < numCols; col++)
            {
                // Calculate the position of the point relative to the center
                float offsetX = (col - (numCols - 1) / 2.0f) * deltaX;
                float offsetY = (row - (numRows - 1) / 2.0f) * deltaY;

                // Calculate the final position of the point
                Vector3 pointPosition = center + new Vector3(offsetX, offsetY, 0);
                currentGrid[row, col] = pointPosition;
            }
        }
        gridPoints.Add(currentGrid);

        foreach (var g in gridPoints)
        {
            for (int row = 0; row < numRows; row++)
            {
                for (int col = 0; col < numCols; col++)
                {
                    spheres[row, col] = AddSphere(g[row, col], radius);
                }
            }
        }
    }

    public GameObject AddSphere(Vector3 pos, float r)
    {
        // ָ������İ뾶
        float radius = r;
        Vector3 position = pos;

        GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        sphere.transform.parent = t; // ���ø�����
        sphere.transform.localScale = new Vector3(radius, radius, radius);
        sphere.transform.position = position;
        // ��ɫ
        sphere.GetComponent<Renderer>().material.color = Color.white;
        return sphere;
    }

    // ����ѡ��״̬�ķ���
    private void HandleSelection(int row, int col)
    {
        if (row == cpyRow && col == cpyCol)
        {
            gazeTimer += Time.deltaTime;

            if (gazeTimer >= gazeDurationThreshold)
            {
                if (selectedSpheres.Count == 0)
                {
                    // ��ѡ�е�������Ϊѡ��״̬
                    spheres[row, col].GetComponent<Renderer>().material.color = Color.red;
                    // ��ѡ�е�list���
                    selectedSpheres.Add(spheres[row, col]);
                    order.Add(row * 3 + col);
                }
                else
                {
                    // ��ֹһ�����ظ�����
                    foreach(var s in selectedSpheres)
                    {
                        if (spheres[row, col].transform.position == s.transform.position)
                            return;
                    }
                    // ��ѡ�е�������Ϊѡ��״̬
                    spheres[row, col].GetComponent<Renderer>().material.color = Color.red;
                    // ��ѡ�е�list���
                    selectedSpheres.Add(spheres[row, col]);
                    order.Add(row * 3 + col);
                    //Debug.Log("row" + row + "col" + col);
                    // ����
                    if (selectedSpheres.Count < 15)
                    {
                        DrawLineBetweenSpheres(selectedSpheres[selectedSpheres.Count - 1], selectedSpheres[selectedSpheres.Count - 2]);

                    }

                }
            }
        }
        // Ŀ�ⲻ��ͬһ�������ü�ʱ��
        else
        {
            gazeTimer = 0f;
        }
    }

    // ����
    private void ResetSpheresColor()
    {
        foreach (var sphere in spheres)
        {
            sphere.GetComponent<Renderer>().material.color = Color.white;
        }

        // ��������� "LineRendererObject" ����������
        if (lineObjects == null) return;
        foreach (GameObject lineObject in lineObjects)
        {
            Destroy(lineObject);
        }

        lineObjects.Clear();
        drawnLines.Clear();
        selectedSpheres.Clear();
        if (order.Count > 2)
        {
            string nameStr = trackingScript.Name + "-" + trackingScript.indexOfSize.ToString() + "-" + trackingScript.indexOfPIN.ToString()
            + "-" + trackingScript.time.ToString();
            string filePath = @"E:\Desktop\data\VRAuth1\PINEntry_" + nameStr + ".txt";
            string OrderedPPIN = "";
            foreach (var o in order) OrderedPPIN = OrderedPPIN + "-" + o.ToString();
            File.WriteAllText(filePath, OrderedPPIN);
        }
        order.Clear();
    }

    private void ResetSpheresLayer()
    {
        foreach (var sphere in spheres)
        {
            Destroy(sphere);
        }
        // ��ʼ�� spheres ����
        spheres = new GameObject[numRows, numCols];
        // 3*3���ĵ�
        Vector3 center = new Vector3(0, highHead, t.position.z);
        // ��ʼ��
        distance = dis[index];
        GenerateSphere(distance, center);
        selectedSpheres.Clear();

        // ��������� "LineRendererObject" ����������
        if (lineObjects == null) return;
        foreach (GameObject lineObject in lineObjects)
        {
            Destroy(lineObject);
        }

        lineObjects.Clear();
        // ����б�
        drawnLines.Clear();
    }

    // ����
    void DrawLineBetweenSpheres(GameObject sphere1, GameObject sphere2)
    {
        // ����һ���µĿ����壬������Ϊ LineRenderer ������
        GameObject lineObject = new GameObject("LineRendererObject");
        lineObjects.Add(lineObject);
        LineRenderer lineRenderer = lineObject.AddComponent<LineRenderer>();

        // ���� LineRenderer ��������ԣ�������ɫ����ȵ�
        lineRenderer.startColor = Color.red;
        lineRenderer.endColor = Color.red;
        lineRenderer.startWidth = 0.05f;
        lineRenderer.endWidth = 0.05f;

        lineRenderer.SetPosition(0, sphere1.transform.position);
        lineRenderer.SetPosition(1, sphere2.transform.position);
        // ����һ���µĲ���
        Material newMaterial = new Material(Shader.Find("Standard"));
        newMaterial.color = Color.red;
        lineRenderer.material = newMaterial;

        // ���´����� LineRenderer ��ӵ��б���
        drawnLines.Add(lineRenderer);
    }

}
