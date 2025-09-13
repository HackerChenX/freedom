# Production Deployment Guide for Freedom Stock Analysis System

## Overview

This guide provides comprehensive instructions for deploying the Freedom Stock Analysis System in a production environment using Docker, Kubernetes, and modern DevOps practices.

## Architecture Overview

### System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   API Gateway   │    │  Microservices  │
│     (Nginx)     │───▶│    (Ingress)    │───▶│   (Pod Group)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │     Cache       │    │    Database     │
│  (Prometheus/   │    │    (Redis)      │    │  (ClickHouse)   │
│   Grafana)      │    │                 │    │    Cluster      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Microservices Components

1. **API Service**: RESTful API endpoints and WebSocket services
2. **Data Processor**: High-performance stock data processing engine
3. **Strategy Analyzer**: Intelligent strategy generation and analysis
4. **Monitor Service**: Real-time market monitoring and alerting
5. **ClickHouse Cluster**: Primary database for stock data storage
6. **Redis**: Caching and session management
7. **Nginx**: Load balancing and reverse proxy

## Prerequisites

### System Requirements

**Minimum Requirements (Development/Testing):**
- CPU: 8 cores
- Memory: 16GB RAM
- Storage: 100GB SSD
- Network: 1Gbps

**Production Requirements:**
- CPU: 32 cores (distributed across nodes)
- Memory: 64GB RAM (distributed across nodes)
- Storage: 1TB SSD (with backup)
- Network: 10Gbps
- Load Balancer: Cloud provider or dedicated hardware

### Software Requirements

- **Kubernetes**: v1.25+ (recommended: v1.28+)
- **Docker**: v20.10+
- **kubectl**: v1.25+
- **kustomize**: v4.5+
- **Helm**: v3.10+ (optional but recommended)

### Cloud Provider Requirements

**AWS:**
- EKS cluster with 3+ worker nodes
- RDS for managed ClickHouse (alternative)
- ElastiCache for Redis (alternative)
- Application Load Balancer
- S3 for backups

**Azure:**
- AKS cluster with 3+ worker nodes
- Azure Database for PostgreSQL (alternative)
- Azure Cache for Redis
- Application Gateway
- Blob Storage for backups

**Google Cloud:**
- GKE cluster with 3+ worker nodes
- Cloud SQL (alternative)
- Memorystore for Redis
- Cloud Load Balancing
- Cloud Storage for backups

## Deployment Process

### 1. Environment Preparation

#### 1.1 Kubernetes Cluster Setup

```bash
# For local development (minikube)
minikube start --cpus=8 --memory=16384 --disk-size=100g

# For production (AWS EKS example)
eksctl create cluster --name freedom-stock --region us-west-2 \
  --nodegroup-name workers --node-type m5.2xlarge --nodes 3

# Verify cluster
kubectl cluster-info
kubectl get nodes
```

#### 1.2 Install Required Tools

```bash
# Install kustomize
curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash

# Install Helm (optional)
curl https://get.helm.sh/helm-v3.12.0-linux-amd64.tar.gz | tar xz
sudo mv linux-amd64/helm /usr/local/bin/

# Install monitoring tools
kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/main/bundle.yaml
```

### 2. Configuration

#### 2.1 Environment Variables

Create environment-specific configuration files:

```bash
# Production environment
cat > .env.prod << EOF
# Database Configuration
CLICKHOUSE_HOST=clickhouse-primary-service
CLICKHOUSE_PORT=8123
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=your_secure_password_here

# Redis Configuration
REDIS_HOST=redis-service
REDIS_PORT=6379
REDIS_PASSWORD=redis_secure_password_here

# Application Configuration
API_PORT=8000
LOG_LEVEL=INFO
PROCESSOR_WORKERS=8
BATCH_SIZE=1000

# Monitoring
GRAFANA_PASSWORD=grafana_admin_password_here
PROMETHEUS_RETENTION_DAYS=30

# Backup Configuration
BACKUP_RETENTION_DAYS=30
S3_BUCKET=your-backup-bucket
S3_PREFIX=freedom-stock-backups
EOF
```

#### 2.2 Kubernetes Secrets

```bash
# Create namespace
kubectl create namespace freedom-stock-prod

# Create secrets
kubectl create secret generic freedom-secrets \
  --from-literal=CLICKHOUSE_PASSWORD=your_secure_password_here \
  --from-literal=CLICKHOUSE_USER=default \
  --from-literal=REDIS_PASSWORD=redis_secure_password_here \
  --from-literal=GRAFANA_PASSWORD=grafana_admin_password_here \
  -n freedom-stock-prod
```

### 3. Database Setup

#### 3.1 ClickHouse Cluster Deployment

```bash
# Apply ClickHouse configuration
kubectl apply -f k8s/base/database-layer.yaml

# Wait for ClickHouse to be ready
kubectl wait --for=condition=Ready pod -l app=clickhouse-primary -n freedom-stock-prod --timeout=300s

# Initialize database schema
kubectl exec -it $(kubectl get pod -l app=clickhouse-primary -n freedom-stock-prod -o jsonpath='{.items[0].metadata.name}') \
  -- clickhouse-client --query "CREATE DATABASE IF NOT EXISTS freedom"
```

#### 3.2 Redis Deployment

```bash
# Redis is included in the database-layer.yaml
# Verify Redis is running
kubectl wait --for=condition=Ready pod -l app=redis -n freedom-stock-prod --timeout=300s
```

### 4. Application Deployment

#### 4.1 Build and Push Container Images

```bash
# Build main application image
docker build -t freedom-stock-analysis:latest .
docker tag freedom-stock-analysis:latest your-registry/freedom-stock-analysis:v1.0.0
docker push your-registry/freedom-stock-analysis:v1.0.0

# Build data processor image
docker build -f Dockerfile.processor -t freedom-data-processor:latest .
docker tag freedom-data-processor:latest your-registry/freedom-data-processor:v1.0.0
docker push your-registry/freedom-data-processor:v1.0.0
```

#### 4.2 Deploy Application Services

```bash
# Use the deployment script
./scripts/deployment/deploy.sh --tag v1.0.0 --environment prod

# Or deploy manually
cd k8s/overlays/prod
kustomize edit set image freedom-stock-analysis=your-registry/freedom-stock-analysis:v1.0.0
kustomize edit set image freedom-data-processor=your-registry/freedom-data-processor:v1.0.0
kubectl apply -k .
```

#### 4.3 Deploy Load Balancer and Ingress

```bash
# Deploy Nginx load balancer
kubectl apply -f k8s/base/load-balancer.yaml

# Wait for load balancer to be ready
kubectl wait --for=condition=Ready pod -l app=nginx-lb -n freedom-stock-prod --timeout=300s
```

### 5. Monitoring Setup

#### 5.1 Deploy Monitoring Stack

```bash
# Deploy Prometheus and Grafana
kubectl apply -f k8s/base/monitoring.yaml

# Wait for monitoring services
kubectl wait --for=condition=Ready pod -l app=prometheus -n freedom-stock-prod --timeout=300s
kubectl wait --for=condition=Ready pod -l app=grafana -n freedom-stock-prod --timeout=300s
```

#### 5.2 Configure Grafana Dashboards

```bash
# Get Grafana admin password
kubectl get secret freedom-secrets -n freedom-stock-prod -o jsonpath='{.data.GRAFANA_PASSWORD}' | base64 --decode

# Port forward to access Grafana
kubectl port-forward svc/grafana-service -n freedom-stock-prod 3000:3000

# Access Grafana at http://localhost:3000
# Import dashboards from config/grafana/dashboards/
```

### 6. Security Hardening

#### 6.1 Apply Security Policies

```bash
# Apply network policies
kubectl apply -f security/network-policies/network-security.yaml

# Apply RBAC configurations
kubectl apply -f security/rbac/rbac-config.yaml

# Apply Pod Security Policies
kubectl apply -f security/policies/security-policies.yaml
```

#### 6.2 Enable Security Scanning

```bash
# Deploy security scanner
kubectl apply -f security/policies/security-policies.yaml

# Schedule regular scans
kubectl create cronjob security-scan --image=aquasec/trivy:latest \
  --schedule="0 2 * * *" \
  -- trivy image --severity HIGH,CRITICAL your-registry/freedom-stock-analysis:latest
```

## Operational Procedures

### Daily Operations

#### Health Monitoring

```bash
# Run health check
./scripts/monitoring/health-monitor.sh

# Check system status
kubectl get pods -n freedom-stock-prod
kubectl get services -n freedom-stock-prod
kubectl top pods -n freedom-stock-prod
```

#### Log Monitoring

```bash
# View application logs
kubectl logs -f deployment/api-service -n freedom-stock-prod

# View system events
kubectl get events -n freedom-stock-prod --sort-by=.metadata.creationTimestamp
```

### Backup and Recovery

#### Daily Backups

```bash
# Run backup script
./scripts/backup/backup.sh

# Schedule backups (add to crontab)
0 2 * * * /path/to/scripts/backup/backup.sh
```

#### Recovery Procedures

```bash
# List available backups
ls -la /backup/freedom-stock/

# Restore from backup
./scripts/backup/backup.sh restore /backup/freedom-stock/20241201-020000.tar.gz
```

### Scaling Operations

#### Horizontal Pod Autoscaler

```bash
# Check current scaling
kubectl get hpa -n freedom-stock-prod

# Manually scale if needed
kubectl scale deployment api-service --replicas=5 -n freedom-stock-prod
```

#### Cluster Scaling

```bash
# AWS EKS scaling
eksctl scale nodegroup --cluster=freedom-stock --nodes=5 workers

# Check node status
kubectl get nodes
kubectl top nodes
```

### Update and Deployment

#### Rolling Updates

```bash
# Update to new version
./scripts/deployment/deploy.sh --tag v1.1.0 --environment prod

# Monitor rollout
kubectl rollout status deployment/api-service -n freedom-stock-prod
```

#### Rollback Procedures

```bash
# Rollback to previous version
./scripts/deployment/deploy.sh --rollback --environment prod

# Or manual rollback
kubectl rollout undo deployment/api-service -n freedom-stock-prod
```

## Performance Tuning

### Database Optimization

#### ClickHouse Tuning

```sql
-- Optimize database settings
ALTER SYSTEM RELOAD CONFIG;
OPTIMIZE TABLE freedom.stock_data;

-- Check performance metrics
SELECT
    database,
    table,
    round(bytes / (1024 * 1024 * 1024), 2) as size_gb,
    rows
FROM system.parts
WHERE database = 'freedom';
```

#### Redis Optimization

```bash
# Check Redis memory usage
kubectl exec -n freedom-stock-prod redis-pod -- redis-cli info memory

# Optimize Redis configuration
kubectl exec -n freedom-stock-prod redis-pod -- redis-cli config set maxmemory-policy allkeys-lru
```

### Application Optimization

#### Resource Allocation

```yaml
# Optimize resource requests and limits
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

#### JVM Tuning (if applicable)

```bash
# Set JVM parameters
export JAVA_OPTS="-Xms2g -Xmx8g -XX:+UseG1GC"
```

## Troubleshooting

### Common Issues

#### Pod Startup Issues

```bash
# Check pod status
kubectl describe pod <pod-name> -n freedom-stock-prod

# Check logs
kubectl logs <pod-name> -n freedom-stock-prod --previous

# Check events
kubectl get events -n freedom-stock-prod --field-selector involvedObject.name=<pod-name>
```

#### Database Connection Issues

```bash
# Test ClickHouse connectivity
kubectl exec -it clickhouse-pod -n freedom-stock-prod -- clickhouse-client --query "SELECT 1"

# Check Redis connectivity
kubectl exec -it redis-pod -n freedom-stock-prod -- redis-cli ping
```

#### Performance Issues

```bash
# Check resource usage
kubectl top pods -n freedom-stock-prod
kubectl top nodes

# Check application metrics
curl http://service-ip:8000/metrics
```

### Emergency Procedures

#### System Outage

1. **Identify the issue**:
   ```bash
   ./scripts/monitoring/health-monitor.sh
   kubectl get pods -n freedom-stock-prod
   ```

2. **Check critical services**:
   ```bash
   kubectl get svc -n freedom-stock-prod
   kubectl get ingress -n freedom-stock-prod
   ```

3. **Restore from backup if needed**:
   ```bash
   ./scripts/backup/backup.sh restore <latest-backup>
   ```

#### Data Corruption

1. **Stop affected services**:
   ```bash
   kubectl scale deployment api-service --replicas=0 -n freedom-stock-prod
   ```

2. **Restore database**:
   ```bash
   # Restore ClickHouse from backup
   kubectl exec -it clickhouse-pod -- clickhouse-client < backup/database/schema.sql
   ```

3. **Restart services**:
   ```bash
   kubectl scale deployment api-service --replicas=3 -n freedom-stock-prod
   ```

## Security Considerations

### Network Security

- All inter-service communication encrypted
- Network policies restrict pod-to-pod communication
- Ingress traffic filtered through WAF

### Data Security

- Database encryption at rest
- SSL/TLS for all external communications
- Regular security scans and vulnerability assessments

### Access Control

- RBAC implemented for all service accounts
- Least privilege principle enforced
- Regular access reviews

## Maintenance Windows

### Planned Maintenance

1. **Pre-maintenance**:
   - Notify stakeholders
   - Create full system backup
   - Prepare rollback procedures

2. **During maintenance**:
   - Scale down non-critical services
   - Apply updates sequentially
   - Monitor system health

3. **Post-maintenance**:
   - Verify all services running
   - Run comprehensive health checks
   - Update documentation

### Emergency Maintenance

- Follow incident response procedures
- Document all changes made
- Conduct post-incident review

## Contact Information

### Support Contacts

- **Primary DevOps**: devops@company.com
- **Database Admin**: dba@company.com
- **Security Team**: security@company.com
- **On-call Engineer**: +1-555-0123

### Escalation Procedures

1. **Level 1**: DevOps Engineer
2. **Level 2**: Senior DevOps/System Architect
3. **Level 3**: CTO/Technical Director

---

**Document Version**: 1.0
**Last Updated**: 2024-12-01
**Next Review**: 2024-03-01