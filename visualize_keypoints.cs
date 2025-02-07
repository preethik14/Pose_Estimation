// Unity script to visualize keypoints on the robot
using System.Collections.Generic;
using UnityEngine;

public class RobotKeypoints : MonoBehaviour
{
    [System.Serializable]
    public class Keypoint
    {
        public string estop;  // Name of the keypoint
        public Transform GameObject;  // Reference to an empty GameObject representing the keypoint
    }

    // public List<Keypoint> keypoints = new List<Keypoint>();  // List of keypoints
    public List<string> keypoints = new List<string> {
        (4.12,-0.706,3.47)};


    private void OnDrawGizmos()
    {
        Gizmos.color = Color.blue;

        foreach (Keypoint kp in keypoints)
        {
            if (kp.GameObject != null)
            {
                Gizmos.DrawSphere(kp.GameObject.position, 0.02f);  // Draw keypoint
                GUIStyle style = new GUIStyle();
                style.normal.textColor = Color.white;
                UnityEditor.Handles.Label(kp.GameObject.position, kp.estop, style); // Show label
            }
        }
    }
}
