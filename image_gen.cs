using UnityEngine;
using UnityEngine.Rendering.PostProcessing;
using System.IO;
using System.Linq;
using System.Collections.Generic;

public class ImageGenerator : MonoBehaviour
{
    public GameObject robotModel;  // Reference to the robot
    public Material quadMaterial;  // Material for the Quad
    public Light sceneLight;       // Main light in the scene
    public string imageFolderPath = "Assets/Backgrounds";  // Image folder path
    public string outputFolderPath = "Assets/GeneratedImages";   // Output folder
   
    public PostProcessProfile blurProfile;  // Post-process profile for blur
    private Texture2D[] backgroundImages;
    private PostProcessVolume postVolume;

    private int imageCounter = 0;
    public string annotationsFolderPath = "Assets/Annotations";  // Folder to store annotation files


    void Start()
    {
        // Load all images from the folder
        LoadBackgroundImages();
        Debug.Log($"Looking for images in: {imageFolderPath}");
       
        // Create post-processing volume
        postVolume = gameObject.AddComponent<PostProcessVolume>();
        postVolume.isGlobal = true;
        postVolume.profile = blurProfile;

        // Start generation process
        StartCoroutine(GenerateImages());
    }

    void LoadBackgroundImages()
    {
    string[] files = Directory.GetFiles(imageFolderPath, "*.jpg");
    backgroundImages = new Texture2D[files.Length];

    for (int i = 0; i < files.Length; i++)
    {
        Debug.Log($"Loading image: {files[i]}"); // Debug log for each image file
        byte[] fileData = File.ReadAllBytes(files[i]);
        Texture2D tex = new Texture2D(2, 2);
        if (tex.LoadImage(fileData))  // Check if the image loaded successfully
        {
            backgroundImages[i] = tex;
            Debug.Log($"Loaded image {i}: {files[i]}");
        }
        else
        {
            Debug.LogError($"Failed to load image: {files[i]}");
        }
    }
    Debug.Log($"Loaded {backgroundImages.Length} background images.");
    }

    System.Collections.IEnumerator GenerateImages()
    {
        foreach (Texture2D bg in backgroundImages)
        {
            for (int i = 0; i < 36; i++)  // 8-10 variations per background
            {
                ApplyBackground(bg);
                RandomizeLighting();
                RotateModel(i);
                ApplyBlur(Random.Range(1.0f, 2.0f));
                yield return new WaitForEndOfFrame();
                CaptureImage();
            }
        }
        Debug.Log("Image Generation Completed!");
    }

    void ApplyBackground(Texture2D texture)
    {
    	if (texture != null)
    	{
        	quadMaterial.mainTexture = texture;
        	Debug.Log("Applied Background.");
        }
        else
        {
        	Debug.Log("Background Texture is null");
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
        Debug.Log("Lighting Randomized.");
    }

    void RotateModel(int step)
    {
        if (robotModel != null)
        {
            // Define the specific steps where you want xRotation
            int[] validSteps = {0, 10, 20, 39, -10, -20, -30};

            // Check if the current step is in the valid steps array
            float xRotation = (validSteps.Contains(step)) ? 10 * step : 0;
            float yRotation = (step % 10 != 0) ? 10 * step : 0;
            float zRotation = 0;  // Assuming zRotation stays at 0 based on your original logic

            // Apply the rotation
            robotModel.transform.rotation = Quaternion.Euler(xRotation, yRotation, zRotation);

            // Debug the actual rotation to confirm it's happening
            Debug.Log($"Rotation: {robotModel.transform.rotation.eulerAngles}");
        }
        else
        {
            Debug.LogWarning("Robot model not assigned!");
        }
    }


    void ApplyBlur(float focusDistance)
    {
        DepthOfField depthOfField;
        if(blurProfile.TryGetSettings(out depthOfField))
        {
        	depthOfField.active = true;
        	depthOfField.focusDistance.value = focusDistance;
        	depthOfField.aperture.value = Random.Range(0.5f, 1.2f);
        	depthOfField.focalLength.value = Random.Range(70f, 100f);
        	Debug.Log("Blur Applied.");
        }
    }

    void CaptureImage()
    {
        string fileName = Path.Combine(outputFolderPath, $"image_{imageCounter:D4}.png");
        ScreenCapture.CaptureScreenshot(fileName);
        SaveAnnotations(imageCounter); 

        imageCounter++;
        Debug.Log($"Captured Image: {fileName}");
    }


    void SaveAnnotations(int step)
    {
        if (robotModel == null)
        {
            Debug.LogWarning("Robot model not assigned, skipping annotation saving.");
            return;
        }

        // Ensure the annotations folder exists
        if (!Directory.Exists(annotationsFolderPath))
        {
            Directory.CreateDirectory(annotationsFolderPath);
        }

        // Get renderer
        Renderer renderer = robotModel.GetComponentInChildren<Renderer>();
        if (renderer == null)
        {
            Debug.LogError("Renderer not found on robotModel or its children!");
            return;
        }

        // Get main camera
        Camera cam = Camera.main;
        if (cam == null)
        {
            Debug.LogError("Main Camera not found!");
            return;
        }

        // Calculate bounding box in image space
            Bounds bounds = renderer.bounds;
        Vector3 worldMin = bounds.min;
        Vector3 worldMax = bounds.max;

        Vector2 screenMin = cam.WorldToViewportPoint(worldMin);
        Vector2 screenMax = cam.WorldToViewportPoint(worldMax);

        // Bounding box properties
        float bboxCentreX = (screenMin.x + screenMax.x) / 2;
        float bboxCentreY = 1 - (screenMin.y + screenMax.y) / 2; // Flip Y-axis
        float bboxWidth = Mathf.Abs(screenMax.x - screenMin.x);
        float bboxHeight = Mathf.Abs(screenMax.y - screenMin.y);

        // Clamp to valid (0,1) range
        bboxCentreX = Mathf.Clamp(bboxCentreX, 0f, 1f);
        bboxCentreY = Mathf.Clamp(bboxCentreY, 0f, 1f);
        bboxWidth = Mathf.Clamp(bboxWidth, 0f, 1f);
        bboxHeight = Mathf.Clamp(bboxHeight, 0f, 1f);

        // Generate scattered keypoints across the entire robot
        int numKeypoints = 30;
        Vector2[] keypoints = new Vector2[numKeypoints];

        for (int i = 0; i < numKeypoints; i++)
        {
            // Randomly sample within the full bounding box
            float randX = Random.Range(bounds.min.x, bounds.max.x);
            float randY = Random.Range(bounds.min.y, bounds.max.y);
            float randZ = Random.Range(bounds.min.z, bounds.max.z);
            
            Vector3 worldPoint = new Vector3(randX, randY, randZ);
            Vector2 screenPoint = cam.WorldToViewportPoint(worldPoint);

            // Flip Y-axis and clamp
            keypoints[i] = new Vector2(
                Mathf.Clamp(screenPoint.x, 0f, 1f),
                Mathf.Clamp(1 - screenPoint.y, 0f, 1f)  // Flip Y for correct viewport mapping
            );
        }

        // Create annotation string
        string annotation = $"0 {bboxCentreX} {bboxCentreY} {bboxWidth} {bboxHeight}";

        // Append keypoints with visibility
        foreach (var keypoint in keypoints)
        {
            annotation += $" {keypoint.x} {keypoint.y} 2";
        }

        // Save to file
        string annotationFilePath = Path.Combine(annotationsFolderPath, $"annotation_{step:D4}.txt");
        File.WriteAllText(annotationFilePath, annotation);

        Debug.Log($"Saved Annotation: {annotationFilePath}");
    }


}

