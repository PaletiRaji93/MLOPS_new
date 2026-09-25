# ArgoCD + Kubernetes — Install Runbook

Companion to [2_DEMO_MASTER_RUNBOOK.md](2_DEMO_MASTER_RUNBOOK.md). Run this once to stand up the
cluster + ArgoCD before Section 4 (CI/CD → GitOps). If the status checks there show the
`gmr-mlops` app or the `mlops` pods missing, set them up here, then go back.

End state: a local minikube cluster running ArgoCD (namespace `argocd`) that auto-deploys
the app stack — gmr-app, postgres, prometheus, grafana — into namespace `mlops`, synced
from `k8s/` on `main`. MLflow runs on the host (not in-cluster).

---

## 1. Prerequisites

> **Already worked through the Installation Guide (`1_MLOps_Installation_Guide.docx`)?**
> You don't need to reinstall anything. Just run that guide's **Verification Checklist**
> section to confirm Docker, kubectl, and minikube are installed and on your PATH — then
> install only whatever it reports as missing, and continue below.

- **Docker Desktop** running.
- **kubectl** — https://kubernetes.io/docs/tasks/tools/
- **minikube** — https://minikube.sigs.k8s.io/docs/start/

```powershell
docker version          # daemon reachable
kubectl version --client
minikube version
```

## 2. Start the cluster

```powershell
minikube start --memory=4096 --cpus=2
minikube addons enable metrics-server
kubectl cluster-info                    # API server reachable
minikube status                         # host/kubelet/apiserver = Running
```

## 3. Start MLflow on the host

The in-cluster app loads its model from MLflow at `host.minikube.internal:5000`, so MLflow
must run on the host (not in the cluster).

```powershell
docker compose up -d mlflow postgres
curl.exe http://localhost:5000/health   # expect: OK
```

## 4. Install ArgoCD

```powershell
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
kubectl wait --for=condition=available deployment/argocd-server -n argocd --timeout=180s
kubectl get pods -n argocd               # all Running (esp. argocd-repo-server, argocd-server)
```

## 5. Get the ArgoCD login (username + password)

- **Username:** `admin` (always)
- **Password:** auto-generated, stored base64-encoded in the `argocd-initial-admin-secret`.
  Decode it:

```powershell
$pw = kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}"
[System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($pw))   # user: admin
```

One-liner version:

```powershell
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | ForEach-Object { [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($_)) }
```

This is the **initial** password. If it was changed in the UI/CLI, the secret no longer
reflects it. Save it somewhere — the secret can be deleted after first login (then the
commands above stop working).

## 6. Create the app namespace + secrets

```powershell
kubectl create namespace mlops

kubectl create secret generic mlops-secrets -n mlops `
  --from-literal=api-key="YOUR_API_KEY" `
  --from-literal=grafana-password="admin123" `
  --from-literal=postgres-user="mlops" `
  --from-literal=postgres-password="mlops" `
  --from-literal=postgres-db="mlops" `
  --dry-run=client -o yaml | kubectl apply -f -
```

Replace `YOUR_API_KEY` with the real key (keep it out of Git — it lives in `.env` locally).

## 7. Deploy the ArgoCD Application

This registers the `gmr-mlops` app — it tells ArgoCD to watch `k8s/` on `main` and auto-sync
(`prune` + `selfHeal`) into namespace `mlops`.

```powershell
kubectl apply -f k8s/argocd-app.yaml
kubectl get applications -n argocd       # gmr-mlops appears -> Synced / Healthy once it rolls out
```

## 8. Access the UIs

Each port-forward blocks — run each in its own terminal tab.

```powershell
kubectl port-forward svc/argocd-server -n argocd 8080:443     # ArgoCD  -> https://localhost:8080 (admin / step 5)
kubectl port-forward svc/gmr-app-svc   -n mlops  8000:8000    # App     -> http://localhost:8000
kubectl port-forward svc/grafana-svc   -n mlops  3000:3000    # Grafana -> http://localhost:3000
kubectl port-forward svc/prometheus-svc -n mlops 9090:9090    # Prom    -> http://localhost:9090
```

## 9. Verify everything

```powershell
kubectl get pods -n mlops                # gmr-app, postgres, prometheus, grafana - all Running
kubectl get svc  -n mlops                # services + ports
kubectl get applications -n argocd       # gmr-mlops: Synced / Healthy
kubectl logs deployment/gmr-app -n mlops --tail=50
```

---

## Troubleshooting

```powershell
# Pod CrashLoopBackOff - why?
kubectl describe pod <pod-name> -n mlops
kubectl logs <pod-name> -n mlops --previous

# App stuck OutOfSync / Unknown - force a sync
kubectl patch application gmr-mlops -n argocd --type merge -p '{\"operation\":{\"sync\":{\"force\":true}}}'

# argocd-repo-server not Running -> app shows Unknown; restart it
kubectl delete pod -n argocd -l app.kubernetes.io/name=argocd-repo-server

# Restart the app deployment
kubectl rollout restart deployment/gmr-app -n mlops
kubectl rollout status  deployment/gmr-app -n mlops

# Nuke and recreate the app stack (keeps ArgoCD)
kubectl delete -f k8s/argocd-app.yaml
kubectl delete namespace mlops
# then redo from step 6
```

> First image pull is slow (gmr-app ~1.8 GB): 3–10 min on a fresh node.

## Cleanup

```powershell
minikube stop                            # stop the cluster (keeps it)
docker compose down                      # stop host MLflow/postgres
minikube delete                          # fully remove the cluster
```

---

## Reference

- `k8s/argocd-app.yaml` — ArgoCD Application: what to watch (`k8s/` on `main`), where to
  deploy (`mlops`), sync policy (`prune` + `selfHeal`).
- `k8s/deployment.yaml` — all app-stack resources (gmr-app, prometheus, grafana, postgres).
- `k8s/namespace.yaml` — the `mlops` namespace.

```
GitHub (main) -> ArgoCD watches k8s/ -> auto-sync -> mlops namespace
  gmr-app (Flask+Gunicorn) -> host.minikube.internal:5000 (MLflow on host)
  prometheus | grafana | postgres
```
