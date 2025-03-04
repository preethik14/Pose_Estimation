using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

public class RobotKeypoints : MonoBehaviour
{
    [System.Serializable]
    public class Keypoint
    {
        public string name;  // Keypoint name
        public Transform transform;  // Empty GameObject for keypoint
    }

    [SerializeField]
    public List<Keypoint> keypoints = new List<Keypoint>();  // List of all keypoints

    private void OnDrawGizmos()
    {
        Gizmos.color = Color.blue;

        foreach (Keypoint kp in keypoints)
        {
            if (kp.transform != null)  // Ensure keypoint exists
            {
                Gizmos.DrawSphere(kp.transform.position, 0.5f);  // Draw sphere at keypoint position
                
                GUIStyle style = new GUIStyle();
                style.normal.textColor = Color.white;
                Handles.Label(kp.transform.position, kp.name, style); // Display name
            }
        }
    }
}