using UnityEngine;
using UnityEngine.Rendering.PostProcessing;
using System.IO;
using System.Collections;
using System.Collections.Generic;


public class ImageGenerator : MonoBehaviour
{
    public GameObject robotModel;  // Reference to the robot
    public Material quadMaterial;  // Material for the Quad (background plane)
    public Light sceneLight;       // Main light in the scene
    public string imageFolderPath = "Assets/Backgrounds";
    public string outputFolderPath = "Assets/GeneratedImages";
    public string annotationsFolderPath = "Assets/Annotations";
    public PostProcessProfile blurProfile;  // Post-processing for blur
    public Camera mainCamera;
    
    private Texture2D[] backgroundImages;
    private PostProcessVolume postVolume;
    private int imageCounter = 0;
    private int imagesPerRotation = 20;
    private int rotationStep = 20;
    
    private List<string> keypointNames = new List<string> {"estop", "power_button", "bumper_rear1", "bumper_rear2", "charge_port", "rear_light_left", "rear_light_right",
    "usb_ports", "lcd_display", "bumper_front_1", "bumper_front_2", "front_light_left", "front_light_right", "right1", "right2", "right3", "right4", "right5", 
    "left1", "left2", "left3", "left4", "left5", "gps", "frame1", "frame2", "frame3", "frame4", "frame5", "frame6"};


    void Start()
    {
        LoadBackgroundImages();
        postVolume = gameObject.AddComponent<PostProcessVolume>();
        postVolume.isGlobal = true;
        postVolume.profile = blurProfile;
        StartCoroutine(GenerateImages());
    }

    void PrintHierarchy(Transform parent, string indent = "")
{
    Debug.Log(indent + parent.name);
    foreach (Transform child in parent)
    {
        PrintHierarchy(child, indent + "  ");
    }
}

    void LoadBackgroundImages()
    {
        string[] files = Directory.GetFiles(imageFolderPath, "*.jpg");
        backgroundImages = new Texture2D[files.Length];

        for (int i = 0; i < files.Length; i++)
        {
            byte[] fileData = File.ReadAllBytes(files[i]);
            Texture2D tex = new Texture2D(2, 2);
            if (tex.LoadImage(fileData))
            {
                backgroundImages[i] = tex;
            }
        }
    }

    IEnumerator GenerateImages()
    {
        foreach (Texture2D bg in backgroundImages)
        {
            ApplyBackground(bg);
            
            for (int i = 0; i < imagesPerRotation; i++)
            {
                RandomizeLighting();
                yield return new WaitForEndOfFrame();
                CaptureImage();
            }
            
            RotateModel();
        }
    }

    void ApplyBackground(Texture2D texture)
    {
        if (texture != null)
        {
            quadMaterial.mainTexture = texture;
        }
    }

    void RandomizeLighting()
    {
        sceneLight.intensity = Random.Range(0.5f, 2.0f);
        sceneLight.color = new Color(
            Random.Range(0.8f, 1f),
            Random.Range(0.8f, 1f),
            Random.Range(0.8f, 1f)
        );
    }

    void RotateModel()
    {
        if (robotModel != null)
        {
            robotModel.transform.Rotate(0, rotationStep, 0);
        }
    }

    void CaptureImage()
    {
        string fileName = Path.Combine(outputFolderPath, $"image_{imageCounter:D4}.png");
        ScreenCapture.CaptureScreenshot(fileName);
        SaveAnnotations(imageCounter);
        imageCounter++;
    }
    void SaveAnnotations(int step)
{
    if (robotModel == null)
    {
        Debug.LogWarning("Robot model not assigned, skipping annotation saving.");
        return;
    }

    if (!Directory.Exists(annotationsFolderPath))
    {
        Directory.CreateDirectory(annotationsFolderPath);
    }

    Camera cam = Camera.main;
    if (cam == null)
    {
        Debug.LogError("Main Camera not found!");
        return;
    }

    // Bounding Box Calculation
    MeshFilter[] meshFilters = robotModel.GetComponentsInChildren<MeshFilter>();
    if (meshFilters.Length == 0)
    {
        Debug.LogError("No MeshFilter found on robotModel!");
        return;
    }

    Vector2 screenMin = new Vector2(1, 1);
    Vector2 screenMax = new Vector2(0, 0);
    List<Vector2> keypoints = new List<Vector2>();

    foreach (MeshFilter meshFilter in meshFilters)
    {
        Mesh mesh = meshFilter.sharedMesh;
        if (mesh == null) continue;

        foreach (Vector3 vertex in mesh.vertices)
        {
            Vector3 worldPos = meshFilter.transform.TransformPoint(vertex);
            Vector3 screenPos = cam.WorldToViewportPoint(worldPos);

            if (screenPos.z > 0)  // Ignore points behind the camera
            {
                screenMin = Vector2.Min(screenMin, new Vector2(screenPos.x, screenPos.y));
                screenMax = Vector2.Max(screenMax, new Vector2(screenPos.x, screenPos.y));
            }
        }
    }

    // Bounding Box in Normalized Coordinates
    float bboxCentreX = (screenMin.x + screenMax.x) / 2;
    float bboxCentreY = 1 - (screenMin.y + screenMax.y) / 2; // Flip Y-axis
    float bboxWidth = Mathf.Abs(screenMax.x - screenMin.x);
    float bboxHeight = Mathf.Abs(screenMax.y - screenMin.y);

    bboxCentreX = Mathf.Clamp(bboxCentreX, 0f, 1f);
    bboxCentreY = Mathf.Clamp(bboxCentreY, 0f, 1f);
    bboxWidth = Mathf.Clamp(bboxWidth, 0f, 1f);
    bboxHeight = Mathf.Clamp(bboxHeight, 0f, 1f);

    Debug.Log($"BBox - Center: ({bboxCentreX}, {bboxCentreY}), Size: ({bboxWidth}, {bboxHeight})");
    PrintHierarchy(robotModel.transform);
    // Keypoint Extraction
    foreach (string keypointName in keypointNames)
    {
        Transform keypointTransform = robotModel.transform.FindDeepChild(keypointName);
        if (keypointTransform != null)
        {
            Debug.LogError($"Keypoint {keypointName} not found in hierarchy!");
            //continue;
            Vector3 worldPos = keypointTransform.position;
            Vector3 screenPos = cam.WorldToViewportPoint(worldPos);

            if (screenPos.z > 0)  // Keypoint is visible
            {
                float normX = Mathf.Clamp(screenPos.x, 0f, 1f);
                float normY = Mathf.Clamp(1 - screenPos.y, 0f, 1f);  // Flip Y-axis
                keypoints.Add(new Vector2(normX, normY));

                Debug.Log($"Keypoint {keypointName} - World: {worldPos}, Screen: {screenPos}, Normalized: ({normX}, {normY}), Visible");
            }
            else
            {
                keypoints.Add(new Vector2(-1f, -1f));  // Mark as not visible
                Debug.LogWarning($"Keypoint {keypointName} is behind the camera! World: {worldPos}, Screen: {screenPos}");
            }
        }
        else
        {
            keypoints.Add(new Vector2(-1f, -1f));  // Keypoint not found
            Debug.LogError($"Keypoint {keypointName} not found in the hierarchy!");
        }
    }

    // Ensure 30 Keypoints (Default Placeholder: -1, -1)
    while (keypoints.Count < 30)
    {
        keypoints.Add(new Vector2(-1f, -1f));
    }

    // Write to File
    string annotationFilePath = Path.Combine(annotationsFolderPath, $"image_{step:D4}.txt");
    using (StreamWriter writer = new StreamWriter(annotationFilePath))
    {
        writer.Write($"0 {bboxCentreX} {bboxCentreY} {bboxWidth} {bboxHeight}");
        foreach (var keypoint in keypoints)
        {
            int visibility = (keypoint.x >= 0 && keypoint.y >= 0) ? 2 : 0;
            writer.Write($" {keypoint.x} {keypoint.y} {visibility}");
        }
    }

    Debug.Log($"Saved Annotation: {annotationFilePath}");
}
}
