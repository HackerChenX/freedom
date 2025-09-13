#!/bin/bash
# Production Deployment Script for Freedom Stock Analysis System
# Version: 1.0
# Author: DevOps Team

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NAMESPACE="freedom-stock-prod"
KUSTOMIZE_OVERLAY="prod"
IMAGE_REGISTRY="ghcr.io"
IMAGE_REPOSITORY="freedom-stock-analysis"
PROCESSOR_REPOSITORY="freedom-data-processor"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Production deployment script for Freedom Stock Analysis System

OPTIONS:
    -t, --tag TAG          Image tag to deploy (required)
    -e, --environment ENV  Environment (prod, staging, dev) [default: prod]
    -c, --kubeconfig PATH  Path to kubeconfig file
    -n, --namespace NS     Kubernetes namespace [default: freedom-stock-prod]
    -d, --dry-run          Perform dry run without applying changes
    -v, --verbose          Enable verbose output
    -h, --help             Show this help message
    --skip-health-check    Skip post-deployment health checks
    --rollback             Rollback to previous version

Examples:
    $0 --tag v1.2.3
    $0 --tag latest --environment staging
    $0 --rollback
    $0 --dry-run --tag v1.2.3

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                case $ENVIRONMENT in
                    prod|staging|dev)
                        KUSTOMIZE_OVERLAY="$ENVIRONMENT"
                        NAMESPACE="freedom-stock-$ENVIRONMENT"
                        ;;
                    *)
                        log_error "Invalid environment: $ENVIRONMENT"
                        exit 1
                        ;;
                esac
                shift 2
                ;;
            -c|--kubeconfig)
                export KUBECONFIG="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                set -x
                shift
                ;;
            --skip-health-check)
                SKIP_HEALTH_CHECK=true
                shift
                ;;
            --rollback)
                ROLLBACK=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    # Validate required arguments
    if [[ -z "${IMAGE_TAG:-}" && -z "${ROLLBACK:-}" ]]; then
        log_error "Image tag is required (--tag) unless performing rollback"
        usage
        exit 1
    fi
}

# Validate prerequisites
validate_prerequisites() {
    log_info "Validating prerequisites..."

    # Check required tools
    local required_tools=("kubectl" "kustomize" "curl" "jq")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is required but not installed"
            exit 1
        fi
    done

    # Check kubectl connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    # Check namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Namespace $NAMESPACE does not exist, creating..."
        kubectl create namespace "$NAMESPACE" || {
            log_error "Failed to create namespace $NAMESPACE"
            exit 1
        }
    fi

    log_success "Prerequisites validated"
}

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."

    # Check cluster resources
    local node_count
    node_count=$(kubectl get nodes --no-headers | grep -c Ready || true)
    if [[ $node_count -lt 3 ]]; then
        log_warning "Less than 3 nodes available. High availability may be affected."
    fi

    # Check persistent volumes
    local pv_count
    pv_count=$(kubectl get pv --no-headers | grep -c Available || true)
    if [[ $pv_count -lt 2 ]]; then
        log_warning "Limited persistent volumes available"
    fi

    # Check current deployment status
    if kubectl get deployment api-service -n "$NAMESPACE" &> /dev/null; then
        local current_replicas
        current_replicas=$(kubectl get deployment api-service -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
        log_info "Current API service replicas: ${current_replicas:-0}"
    fi

    log_success "Pre-deployment checks completed"
}

# Backup current deployment
backup_current_deployment() {
    log_info "Creating backup of current deployment..."

    local backup_dir="$PROJECT_ROOT/backups/$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"

    # Backup deployment configurations
    kubectl get deployments -n "$NAMESPACE" -o yaml > "$backup_dir/deployments.yaml" || true
    kubectl get services -n "$NAMESPACE" -o yaml > "$backup_dir/services.yaml" || true
    kubectl get configmaps -n "$NAMESPACE" -o yaml > "$backup_dir/configmaps.yaml" || true
    kubectl get secrets -n "$NAMESPACE" -o yaml > "$backup_dir/secrets.yaml" || true

    # Save current image versions
    kubectl get deployments -n "$NAMESPACE" -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.template.spec.containers[*].image}{"\n"}{end}' > "$backup_dir/current_images.txt" || true

    log_success "Backup created at: $backup_dir"
    echo "$backup_dir" > /tmp/freedom_backup_path
}

# Deploy application
deploy_application() {
    log_info "Deploying Freedom Stock Analysis System..."

    cd "$PROJECT_ROOT/k8s/overlays/$KUSTOMIZE_OVERLAY"

    if [[ -n "${IMAGE_TAG:-}" ]]; then
        # Update image tags in kustomization
        log_info "Updating image tags to $IMAGE_TAG..."
        kustomize edit set image "$IMAGE_REPOSITORY=$IMAGE_REGISTRY/$IMAGE_REPOSITORY:$IMAGE_TAG"
        kustomize edit set image "$PROCESSOR_REPOSITORY=$IMAGE_REGISTRY/$PROCESSOR_REPOSITORY:$IMAGE_TAG"
    fi

    # Apply configuration
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        log_info "Dry run mode - showing what would be applied:"
        kustomize build . | kubectl apply --dry-run=client -f -
        return 0
    fi

    log_info "Applying Kubernetes manifests..."
    kustomize build . | kubectl apply -f -

    log_success "Application deployed"
}

# Wait for deployment rollout
wait_for_rollout() {
    log_info "Waiting for deployment rollout to complete..."

    local deployments=("api-service" "data-processor" "strategy-analyzer" "monitor-service")
    local timeout=900  # 15 minutes

    for deployment in "${deployments[@]}"; do
        log_info "Waiting for $deployment rollout..."
        if ! kubectl rollout status deployment "$deployment" -n "$NAMESPACE" --timeout="${timeout}s"; then
            log_error "Rollout failed for $deployment"
            return 1
        fi
    done

    log_success "All deployments rolled out successfully"
}

# Health checks
run_health_checks() {
    if [[ "${SKIP_HEALTH_CHECK:-false}" == "true" ]]; then
        log_info "Skipping health checks"
        return 0
    fi

    log_info "Running post-deployment health checks..."

    # Wait for pods to be ready
    log_info "Waiting for pods to be ready..."
    kubectl wait --for=condition=Ready pod -l app=api-service -n "$NAMESPACE" --timeout=300s

    # Get service endpoint
    local service_ip
    service_ip=$(kubectl get svc nginx-lb-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")

    if [[ -z "$service_ip" ]]; then
        # Try NodePort or ClusterIP
        service_ip=$(kubectl get svc nginx-lb-service -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
        local service_port
        service_port=$(kubectl get svc nginx-lb-service -n "$NAMESPACE" -o jsonpath='{.spec.ports[0].port}')
        local service_url="http://$service_ip:$service_port"
    else
        local service_url="http://$service_ip"
    fi

    # Health endpoint checks
    log_info "Checking health endpoint: $service_url/health"
    if curl -f -s --max-time 30 "$service_url/health" > /dev/null; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        return 1
    fi

    # Info endpoint check
    log_info "Checking info endpoint: $service_url/info"
    if curl -f -s --max-time 30 "$service_url/info" | jq -r '.system' | grep -q "股票分析系统"; then
        log_success "Info endpoint check passed"
    else
        log_error "Info endpoint check failed"
        return 1
    fi

    # Database connectivity check
    log_info "Checking database connectivity..."
    local api_pod
    api_pod=$(kubectl get pod -l app=api-service -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}')

    if kubectl exec -n "$NAMESPACE" "$api_pod" -- python -c "
import sys
sys.path.append('/app')
from db.enhanced_connection_pool import ClickHouseConnectionPool
try:
    pool = ClickHouseConnectionPool()
    print('Database connection successful')
except Exception as e:
    print(f'Database connection failed: {e}')
    sys.exit(1)
"; then
        log_success "Database connectivity check passed"
    else
        log_error "Database connectivity check failed"
        return 1
    fi

    log_success "All health checks passed"
}

# Rollback deployment
rollback_deployment() {
    log_info "Rolling back deployment..."

    local deployments=("api-service" "data-processor" "strategy-analyzer" "monitor-service")

    for deployment in "${deployments[@]}"; do
        if kubectl get deployment "$deployment" -n "$NAMESPACE" &> /dev/null; then
            log_info "Rolling back $deployment..."
            kubectl rollout undo deployment "$deployment" -n "$NAMESPACE"
        fi
    done

    # Wait for rollback to complete
    wait_for_rollout

    log_success "Rollback completed"
}

# Post-deployment tasks
post_deployment_tasks() {
    log_info "Running post-deployment tasks..."

    # Update monitoring dashboards
    if kubectl get configmap grafana-dashboards -n "$NAMESPACE" &> /dev/null; then
        log_info "Updating Grafana dashboards..."
        kubectl rollout restart deployment/grafana -n "$NAMESPACE" || true
    fi

    # Restart monitoring to pick up new services
    if kubectl get deployment prometheus -n "$NAMESPACE" &> /dev/null; then
        log_info "Restarting Prometheus to discover new targets..."
        kubectl rollout restart deployment/prometheus -n "$NAMESPACE" || true
    fi

    # Clear application caches
    log_info "Clearing application caches..."
    kubectl delete pod -l app=redis -n "$NAMESPACE" --grace-period=10 || true

    log_success "Post-deployment tasks completed"
}

# Deployment summary
deployment_summary() {
    log_info "Deployment Summary"
    echo "===================="
    echo "Environment: $KUSTOMIZE_OVERLAY"
    echo "Namespace: $NAMESPACE"
    echo "Image Tag: ${IMAGE_TAG:-'rollback'}"
    echo "Timestamp: $(date)"
    echo ""

    # Show deployment status
    kubectl get deployments -n "$NAMESPACE" -o wide

    echo ""
    # Show service endpoints
    kubectl get services -n "$NAMESPACE" -o wide

    log_success "Deployment completed successfully!"
}

# Error handling
error_handler() {
    local exit_code=$?
    log_error "Deployment failed with exit code $exit_code"

    # If we have a backup, offer to rollback
    if [[ -f /tmp/freedom_backup_path ]]; then
        log_warning "Backup available. Consider running rollback."
    fi

    exit $exit_code
}

# Main function
main() {
    log_info "Starting Freedom Stock Analysis System deployment..."

    # Set error handler
    trap error_handler ERR

    # Parse arguments
    parse_args "$@"

    # Handle rollback
    if [[ "${ROLLBACK:-false}" == "true" ]]; then
        validate_prerequisites
        rollback_deployment
        run_health_checks
        deployment_summary
        exit 0
    fi

    # Normal deployment flow
    validate_prerequisites
    pre_deployment_checks
    backup_current_deployment
    deploy_application
    wait_for_rollout
    run_health_checks
    post_deployment_tasks
    deployment_summary
}

# Run main function with all arguments
main "$@"