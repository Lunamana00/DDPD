using System;

[Serializable]
public struct EgoMotionSample
{
    public float forward;
    public float right;
    public float yaw;

    public EgoMotionSample(float forward, float right, float yaw)
    {
        this.forward = forward;
        this.right = right;
        this.yaw = yaw;
    }
}

[Serializable]
public struct PathPoint
{
    public float forward;
    public float right;
}

[Serializable]
public class PathPredictionResponse
{
    public int future_steps;
    public int selected_mode;
    public PathPoint[] path;
    public float[] mode_confidences;
}

public static class PathPredictionResponseExtensions
{
    public static bool HasPath(this PathPredictionResponse response)
    {
        return response != null && response.path != null && response.path.Length > 0;
    }
}
