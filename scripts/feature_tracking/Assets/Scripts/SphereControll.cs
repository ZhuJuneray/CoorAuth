using UnityEngine;

public class SphereControll : MonoBehaviour
{
    private GameObject[] spheres;  // 用于存储四个球体的数组
    private int currentIndex = 0;  // 当前显示的球体索引
    private bool begin, init;
    private AuthController trackingScript;

    void Start()
    {
        trackingScript = GameObject.Find("Controller").GetComponent<AuthController>();//获取controller
        this.begin = trackingScript.Begin;
        this.init = true;
        // 获取Sphere下的所有子物体球体
        spheres = new GameObject[transform.childCount];
        for (int i = 0; i < transform.childCount; i++)
        {
            spheres[i] = transform.GetChild(i).gameObject;
        }

    }

    private void Update()
    {
        this.begin = trackingScript.Begin;
        if (begin && init)
        {
            foreach (var s in spheres)
            {
                s.SetActive(false);
            }
            // 获取第一个球体的Renderer组件
            Renderer currentRenderer = spheres[0].GetComponent<Renderer>();
            currentRenderer.material.color = Color.red;
            spheres[0].SetActive(true);
            // 启动定时器，每隔一秒调用ShowNextSphere方法
            InvokeRepeating("ShowNextSphere", 2f, 2f);
            init = false;
        }
    }

    void ShowNextSphere()
    {
        
        spheres[currentIndex].SetActive(false);
        currentIndex = (currentIndex + 1) % spheres.Length;
        spheres[currentIndex].SetActive(true);
        // 获取当前球体的Renderer组件
        Renderer currentRenderer = spheres[currentIndex].GetComponent<Renderer>();
        currentRenderer.material.color = Color.red;
    }
}
