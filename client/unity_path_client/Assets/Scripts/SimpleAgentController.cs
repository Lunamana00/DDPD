using UnityEngine;

public class SimpleAgentController : MonoBehaviour
{
    [SerializeField] private float moveSpeed = 2.0f;
    [SerializeField] private float turnSpeed = 95.0f;
    [SerializeField] private bool autoDriveWhenIdle = true;
    [SerializeField] private float autoDriveTurnSpeed = 18.0f;

    private void Update()
    {
        float move = Input.GetAxisRaw("Vertical");
        float turn = Input.GetAxisRaw("Horizontal");
        bool hasInput = Mathf.Abs(move) > 0.01f || Mathf.Abs(turn) > 0.01f;

        if (!hasInput && autoDriveWhenIdle)
        {
            move = 1.0f;
            turn = Mathf.Sin(Time.time * 0.6f) * 0.35f;
        }

        float appliedTurnSpeed = hasInput ? turnSpeed : autoDriveTurnSpeed;
        transform.Rotate(Vector3.up, turn * appliedTurnSpeed * Time.deltaTime, Space.World);
        transform.position += transform.forward * (move * moveSpeed * Time.deltaTime);
    }
}
