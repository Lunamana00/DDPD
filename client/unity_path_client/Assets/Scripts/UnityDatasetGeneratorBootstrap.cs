using System;
using UnityEngine;

public static class UnityDatasetGeneratorBootstrap
{
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    private static void CreateGeneratorForBatchMode()
    {
        string[] args = Environment.GetCommandLineArgs();
        bool shouldGenerate = false;
        foreach (string arg in args)
        {
            if (arg == "--ddpd-generate-dataset")
            {
                shouldGenerate = true;
                break;
            }
        }

        if (!shouldGenerate)
        {
            return;
        }

        GameObject generatorObject = new GameObject("DDPD Unity Dataset Generator");
        UnityDatasetGenerator generator = generatorObject.AddComponent<UnityDatasetGenerator>();
        generator.GenerateFromContextMenu();
    }
}
