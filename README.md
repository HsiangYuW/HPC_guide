# HPC 指南

> 指令中的 `<ACCOUNT>` 請替換成你自己的帳號(本文範例以 `csliao` 示意)。

---

## 0. 前情提要

- **叢集 = 一堆機器 + 一個排隊系統(Slurm)+ 一個共用倉庫(`/storage`)。**
- **不是**直接在的 GPU 機器上打指令;你在**登入節點**寫好腳本,用 `sbatch` **排隊**,Slurm 把運算節點配給你。
- 檔案放在**共用網路儲存 `/storage`**,所有節點都看得到 → 換到哪台節點,資料都在。
- 軟體環境靠 **Apptainer 容器**提供。

```
本機 ──ssh──▶ 登入節點(A12-01, x86)──sbatch──▶ Slurm 排程 ──▶ 運算節點(GPU/CPU)
                        │                                              │
                        └──────────── 共用儲存 /storage ───────────────┘
```

---

## 1. 登入

```bash
ssh <ACCOUNT>@140.112.170.43 -p 2201
```
- 進來後你在**登入節點 `A12-01`**(x86_64)。用 `hostname` 可確認。
- 登入節點只用來:**編輯檔案、傳檔(rsync/scp)、提交任務(sbatch)、看結果**。
- **登入節點非跑訓練/大量運算的地方**。

傳檔範例(本機 → HPC):
```bash
rsync -avz -e 'ssh -p 2201' 本地檔案 <ACCOUNT>@140.112.170.43:~/目標路徑/
```

---

## 2. 節點與分區(硬體長怎樣)

### 節點角色
| 類型 | 例子 | 架構 | 用途 |
|---|---|---|---|
| 登入節點 | `A12-01` | x86_64 | 進入點,不跑運算 |
| CPU 運算節點 | `A09-xx ~ A12-xx` | x86_64 | 純 CPU 計算(144 核、~770GB RAM) |
| GPU 運算節點 | `gpu1 ~ gpu4` | **aarch64（ARM）** | **NVIDIA GH200**(每台 1 顆) |
| GPU 運算節點 | `amd` | **x86_64** | 有 4 顆 GPU,但架構不同(見下方坑) |

### 分區(partition)= 不同排隊窗口
| 分區 | 時間上限 | 節點 |
|---|---|---|
| `standard` / `long` / `short` | 各有不同 | CPU 節點群 |
| `gpu_long` | **3 天** | `gpu1`(GH200/ARM)+ `amd`(x86) |
| `gpu_short` | **12 小時** | `gpu2` / `gpu3` / `gpu4`(GH200/ARM) |

### 注意架構(重要)
- GPU 節點 `gpu1~4` 的 CPU 是 **ARM(aarch64)**(GH200 = Grace ARM + Hopper GPU 整合晶片)。
- `amd` 節點是 **x86_64**。
- **如果你的容器是 arm64 編譯的,只能跑在 `gpu1~4`,不能跑在 `amd`**。用 `--nodelist=gpu1`(或 gpu2/3/4)強制指定,否則排到 `amd` 會報:
  ```
  the image's architecture (arm64) could not run on the host's (amd64)
  ```

### 查硬體的指令
```bash
sinfo -N -l                     # 所有節點一覽(狀態/CPU/記憶體/分區)
scontrol show node gpu1         # 單一節點詳細(CPU、記憶體、Gres=GPU、Arch)
```

---

## 3. 儲存與路徑

- **`/storage` 是共用網路儲存**(NFS),所有節點掛載同一份 → 換節點檔案都在。
- 你的家目錄 `$HOME` 在 `/storage/...` 底下(例如 `/storage/professor/<ACCOUNT>`)。(e.g /storage/professor/csliao)
- ⚠️ 節點本機的 `/tmp`、`/var/spool` **換節點就消失**,只放單次暫存。**要保留的東西(程式碼、資料、checkpoint、log)一律放 `/storage`。**
- 有**容量配額**(依身分不同,例如 大學部 1TB / 碩士 2TB / 博士 3TB)。查用量:
  ```bash
  du -sh ~
  ```

---

## 4. Slurm 基本操作

| 指令 | 作用 |
|---|---|
| `sbatch 腳本.sh` | 提交批次任務去排隊 |
| `squeue -u <ACCOUNT>` | 看自己的任務(`R`=跑中,`PD`=排隊) |
| `scancel <jobid>` | 取消任務 |
| `sinfo` | 看節點/分區狀態 |
| `sacct -u <ACCOUNT>` | 看歷史任務(狀態、耗時、結束碼) |
| `srun ...` | 互動式直接在運算節點跑一段指令 |

看即時 log(SBATCH 的 `--output` 指定的檔案):
```bash
tail -f logs/xxx_<jobid>.out
```

---

## 5. 容器 Apptainer(軟體環境從哪來)

### 為什麼要容器
運算節點系統很精簡,**沒有預裝 PyTorch 等套件,你也沒有 root 不能亂裝**。解法是用 **Apptainer**(HPC 版 Docker,免 root)進入一個「打包好的軟體環境盒」。

### 管理員預建的映像檔
- 位置(範例):`/storage/software/containers/pytorch-ngc-aarch64.sif`
- 這是 NVIDIA NGC PyTorch 容器的 **ARM(aarch64)版**,內含:**PyTorch、CUDA、cuDNN、flash-attn、Python** 等一整套。
- ⚠️ **這是 arm64 映像 → 只能在 `gpu1~4`(aarch64)開,不能在 `amd`。**

### 「不重建 image」,而是「疊加」
> **重點觀念:不修改、也不重建那個 `.sif`(唯讀、沒 root權限可以改)。**
> 要額外的套件時,用 **`pip install --user`** 裝到自己家目錄的一個資料夾,疊在容器之上:

```bash
export PYTHONUSERBASE=$HOME/my_pylibs        # 存放額外套件的目錄
apptainer exec --nv \
  --env PYTHONUSERBASE="$PYTHONUSERBASE" \
  --env PYTHONPATH="$PYTHONUSERBASE/lib/python3.10/site-packages" \
  /storage/software/containers/pytorch-ngc-aarch64.sif \
  pip install --user 你要的套件
```
- 容器內建的 torch / CUDA / flash-attn **保持不動**。
- 你的額外套件疊在 `$PYTHONUSERBASE`,匯入時優先於容器內建。
- **不需另外 build 一個 image** 一般使用者用「官方 image + `--user` 疊加」即可;

### 在容器內執行程式的通式
```bash
apptainer exec --nv <映像檔.sif> python 你的程式.py
```
- `--nv`:把主機的 NVIDIA 驅動/GPU 掛進容器。
- 容器**不換檔案系統** → 裡面一樣看得到 `/storage`(軟體來自容器,資料來自主機)。

---

## 6. 怎麼指定「用哪個硬體」

在 SBATCH 標頭(或 srun 參數)用這幾個旗標:

| 旗標 | 意義 |
|---|---|
| `--partition=<名稱>` | 選分區(決定時間上限與可用節點) |
| `--nodelist=<節點>` | **指定節點**(用來對齊架構,如 `gpu1`) |
| `--gres=gpu:1` | **要幾顆 GPU**(不寫就分不到 GPU) |


> ⚠️ 常見錯誤:忘了 `--gres=gpu:1`(拿不到 GPU)、忘了 `--account`(被拒)、或架構沒對齊(arm 容器排到 `amd`)。

---

## 7. Shell Script

### 範例腳本
用來確認容器 + GPU 正常:
```bash
#!/bin/bash
#SBATCH --job-name=smoke
#SBATCH --output=logs/smoke_%j.out
#SBATCH --partition=gpu_short
#SBATCH --nodelist=gpu2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --account=<ACCOUNT>

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"; mkdir -p logs
APP=/opt/hpc/apptainer/bin/apptainer
SIF=/storage/software/containers/pytorch-ngc-aarch64.sif
"$APP" exec --nv "$SIF" python -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
```bash
sbatch smoke.sh && squeue -u <ACCOUNT>
```
---

## 8. 常見錯誤

| 症狀 | 原因 | 解法 |
|---|---|---|
| `image's architecture (arm64) could not run on host's (amd64)` | arm 容器排到 `amd`(x86) | `--nodelist=gpu1`(或 gpu2~4) |
| `./env.sh: No such file` / `command not found` / job 秒結束 | Slurm 跑 spool 副本,`$0`/`./` 指錯 | 用 `cd "$SLURM_SUBMIT_DIR"` |
| 程式找不到 GPU / `CUDA available: False` | 忘了 `--gres=gpu:1` 或沒 `--nv` | 兩者都補上 |
| 任務被拒 | 沒 `--account` | 加 `--account=<ACCOUNT>` |
| 在運算節點跑 `sbatch`/`scancel` 失敗 | Slurm 指令是 x86,aarch64 節點不能執行 | 排程動作只在**登入節點**做 |
| 換節點檔案不見 | 放到本機 `/tmp` 了 | 一律放 `/storage`(家目錄) |
| 裝套件把容器的 torch 弄壞 | 覆蓋了容器內建 | 只用 `pip install --user`,別動容器 torch |


---

