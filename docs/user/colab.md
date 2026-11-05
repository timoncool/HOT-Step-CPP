# Running on Google Colab

If you have no suitable GPU, [`colab/HOT-Step.ipynb`](../../colab/HOT-Step.ipynb) runs the
Linux release on a Colab GPU and gives you a link to the normal web UI. The free T4 runs ACE-Step 1.5 and YuE2.

[Open in Colab](https://colab.research.google.com/github/scragnog/HOT-Step-CPP/blob/master/colab/HOT-Step.ipynb)

## Workflow

1. Open the notebook and set **Runtime > Change runtime type** to a GPU.
2. Run the cells in order. Each one has a small form with its settings.

| Cell | What it does |
|---|---|
| 1. Install | Downloads the `linux-x64-cuda12.8` archive of the chosen release (`latest` by default), unless that version is already installed. Adds the CUDA 12 runtime and cuBLAS from pip when the engine can't find them on the Colab image. |
| 2. Google Drive | Optional. Puts `models/` and `adapters/` on your Drive under `HOT-Step/`, so they download once. Restores your songs and training data from the last backup. |
| 3. Models | Downloads one pack from the app's model registry: `quick-start`, `minimal` or `xl-quality` for ACE-Step 1.5, or a `yue2-*` pack (`yue2-recommended` by default). `none` skips it. Files already present at the right size are skipped. A Hugging Face token is optional and only speeds up downloads. |
| 4. Start | Launches the server through `HOT-Step.sh` and waits for `/api/health`. Prints the end of the log if the engine fails to start. |
| 5. Open the app | Prints the link. `colab` uses Colab's own port proxy: no account, and only your signed-in browser can open it. `ngrok` gives a public link and needs a free authtoken. |
| 6. Back up | Copies your songs, the library database, training datasets and runs to `data/` in the Drive folder. |

<!-- TODO(verify): the notebook predates the v1.4.1 data folder. With a current release, HOT-Step-Data sits at /content/HOT-Step-Data: the first start moves the Drive-linked models/ and adapters/ off Drive into it, later downloads land there, and cells 2 and 6 still restore and back up server/data, which no longer holds the library. Fix the notebook, then recheck cells 2, 3 and 6 and the adapters and Model Manager tips below. -->

## Tips and limits

- **Run cell 6 before the session ends.** The library database stays on the Colab disk while you
  work, because SQLite doesn't run reliably on Drive. Anything not backed up is lost when Colab
  recycles the machine.
- **An ngrok link is public.** Anyone who has it can use your session until it ends.
- **Why no Cloudflare quick tunnel:** quick tunnels
  [don't support Server-Sent Events](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/),
  so the terminal, download progress and the assistant's streamed replies stay empty through one.
- **Use Q8_0 models on a T4.** The T4 has no fast BF16 path, so BF16 files run slowly. The
  language models and DiTs in the cell 3 packs are Q8_0 or smaller, except
  `yue2-full-precision`, which is BF16.
- **Add your own adapters** by copying them into `HOT-Step/adapters/` on Drive before you run
  cell 2. Download more models from the [Model Manager](studios/model-manager.md) in the UI;
  with Drive enabled they also persist.
- **Why `cuda12.8` and not `cuda13.1`:** the CUDA 13 build needs a newer NVIDIA driver than
  Colab is guaranteed to have. The CUDA 12 build runs on any Colab GPU.
- **Restart works:** after **Restart** in the sidebar, `HOT-Step.sh` relaunches the server.
  If the page stops responding, run cell 4 again. The link from cell 5 stays the same.
- **MiniMax-Music3** packs are not in the cell 3 list, because they haven't been tried on a
  T4. You can still download them from the Model Manager.

## Related

- [Getting started](getting-started.md)
- [Model files](models.md)
- [Troubleshooting](troubleshooting.md)
