#!/bin/bash
# System Health Monitoring Script for Freedom Stock Analysis System
# Version: 1.0

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${NAMESPACE:-freedom-stock-prod}"
ALERT_WEBHOOK="${ALERT_WEBHOOK:-}"
LOG_FILE="${LOG_FILE:-/var/log/freedom-stock-monitor.log}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
    log "INFO: $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    log "SUCCESS: $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    log "WARNING: $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    log "ERROR: $1"
}

# Send alert function
send_alert() {
    local severity="$1"
    local message="$2"

    if [[ -n "$ALERT_WEBHOOK" ]]; then
        curl -X POST "$ALERT_WEBHOOK" \
             -H "Content-Type: application/json" \
             -d "{\"text\":\"[$severity] Freedom Stock System: $message\"}" \
             2>/dev/null || log_warning "Failed to send webhook alert"
    fi

    # Log critical alerts
    if [[ "$severity" == "CRITICAL" ]]; then
        logger -p user.crit "Freedom Stock System: $message"
    fi
}

# Check Kubernetes cluster connectivity
check_cluster_connectivity() {
    log_info "Checking Kubernetes cluster connectivity..."

    if kubectl cluster-info &>/dev/null; then
        log_success "Cluster connectivity OK"
        return 0
    else
        log_error "Cannot connect to Kubernetes cluster"
        send_alert "CRITICAL" "Cannot connect to Kubernetes cluster"
        return 1
    fi
}

# Check namespace and pods
check_pods_status() {
    log_info "Checking pods status in namespace $NAMESPACE..."

    if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
        log_error "Namespace $NAMESPACE does not exist"
        send_alert "CRITICAL" "Namespace $NAMESPACE does not exist"
        return 1
    fi

    local failed_pods=()
    local not_ready_pods=()

    # Get all pods in the namespace
    while IFS= read -r line; do
        local pod_name=$(echo "$line" | awk '{print $1}')
        local ready=$(echo "$line" | awk '{print $2}')
        local status=$(echo "$line" | awk '{print $3}')
        local restarts=$(echo "$line" | awk '{print $4}')

        if [[ "$status" != "Running" && "$status" != "Completed" ]]; then
            failed_pods+=("$pod_name:$status")
        fi

        if [[ "$ready" == *"/"* ]]; then
            local ready_count=$(echo "$ready" | cut -d'/' -f1)
            local total_count=$(echo "$ready" | cut -d'/' -f2)
            if [[ "$ready_count" != "$total_count" ]]; then
                not_ready_pods+=("$pod_name:$ready")
            fi
        fi

        # Check for excessive restarts
        if [[ "$restarts" =~ ^[0-9]+$ ]] && [[ "$restarts" -gt 5 ]]; then
            log_warning "Pod $pod_name has $restarts restarts"
            send_alert "WARNING" "Pod $pod_name has excessive restarts: $restarts"
        fi

    done < <(kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null)

    if [[ ${#failed_pods[@]} -gt 0 ]]; then
        log_error "Failed pods detected: ${failed_pods[*]}"
        send_alert "CRITICAL" "Failed pods: ${failed_pods[*]}"
        return 1
    fi

    if [[ ${#not_ready_pods[@]} -gt 0 ]]; then
        log_warning "Not ready pods detected: ${not_ready_pods[*]}"
        send_alert "WARNING" "Not ready pods: ${not_ready_pods[*]}"
    fi

    log_success "All pods are healthy"
    return 0
}

# Check deployments status
check_deployments_status() {
    log_info "Checking deployments status..."

    local critical_deployments=("api-service" "clickhouse-primary" "redis")
    local failed_deployments=()

    for deployment in "${critical_deployments[@]}"; do
        if ! kubectl get deployment "$deployment" -n "$NAMESPACE" &>/dev/null; then
            failed_deployments+=("$deployment:NOT_FOUND")
            continue
        fi

        local desired=$(kubectl get deployment "$deployment" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
        local available=$(kubectl get deployment "$deployment" -n "$NAMESPACE" -o jsonpath='{.status.availableReplicas}')

        available=${available:-0}

        if [[ "$available" -lt "$desired" ]]; then
            failed_deployments+=("$deployment:$available/$desired")
        fi
    done

    if [[ ${#failed_deployments[@]} -gt 0 ]]; then
        log_error "Deployment issues detected: ${failed_deployments[*]}"
        send_alert "CRITICAL" "Deployment issues: ${failed_deployments[*]}"
        return 1
    fi

    log_success "All deployments are healthy"
    return 0
}

# Check service endpoints
check_service_endpoints() {
    log_info "Checking service endpoints..."

    # Get load balancer IP or service IP
    local service_ip=$(kubectl get svc nginx-lb-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")

    if [[ -z "$service_ip" ]]; then
        service_ip=$(kubectl get svc nginx-lb-service -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
        local service_port=$(kubectl get svc nginx-lb-service -n "$NAMESPACE" -o jsonpath='{.spec.ports[0].port}')
        local base_url="http://$service_ip:$service_port"
    else
        local base_url="http://$service_ip"
    fi

    # Health check
    if curl -f -s --max-time 10 "$base_url/health" >/dev/null; then
        log_success "Health endpoint responding"
    else
        log_error "Health endpoint not responding"
        send_alert "CRITICAL" "Health endpoint not responding: $base_url/health"
        return 1
    fi

    # API info check
    if curl -f -s --max-time 10 "$base_url/info" | grep -q "股票分析系统"; then
        log_success "API info endpoint responding"
    else
        log_error "API info endpoint not responding properly"
        send_alert "CRITICAL" "API info endpoint not responding properly"
        return 1
    fi

    return 0
}

# Check database connectivity
check_database_connectivity() {
    log_info "Checking database connectivity..."

    # Get ClickHouse pod
    local clickhouse_pod=$(kubectl get pod -l app=clickhouse-primary -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [[ -z "$clickhouse_pod" ]]; then
        log_error "ClickHouse pod not found"
        send_alert "CRITICAL" "ClickHouse pod not found"
        return 1
    fi

    # Test ClickHouse connectivity
    if kubectl exec -n "$NAMESPACE" "$clickhouse_pod" -- clickhouse-client --query "SELECT 1" &>/dev/null; then
        log_success "ClickHouse database responding"
    else
        log_error "ClickHouse database not responding"
        send_alert "CRITICAL" "ClickHouse database not responding"
        return 1
    fi

    # Check Redis connectivity
    local redis_pod=$(kubectl get pod -l app=redis -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [[ -n "$redis_pod" ]]; then
        if kubectl exec -n "$NAMESPACE" "$redis_pod" -- redis-cli ping | grep -q "PONG"; then
            log_success "Redis responding"
        else
            log_error "Redis not responding"
            send_alert "CRITICAL" "Redis not responding"
            return 1
        fi
    fi

    return 0
}

# Check resource usage
check_resource_usage() {
    log_info "Checking resource usage..."

    # Check node resource usage
    while IFS= read -r line; do
        local node_name=$(echo "$line" | awk '{print $1}')
        local cpu_usage=$(echo "$line" | awk '{print $2}' | sed 's/%//')
        local memory_usage=$(echo "$line" | awk '{print $4}' | sed 's/%//')

        if [[ "$cpu_usage" =~ ^[0-9]+$ ]] && [[ "$cpu_usage" -gt 85 ]]; then
            log_warning "Node $node_name CPU usage high: $cpu_usage%"
            send_alert "WARNING" "Node $node_name CPU usage high: $cpu_usage%"
        fi

        if [[ "$memory_usage" =~ ^[0-9]+$ ]] && [[ "$memory_usage" -gt 85 ]]; then
            log_warning "Node $node_name memory usage high: $memory_usage%"
            send_alert "WARNING" "Node $node_name memory usage high: $memory_usage%"
        fi

    done < <(kubectl top nodes --no-headers 2>/dev/null || echo "")

    # Check pod resource usage
    local high_cpu_pods=()
    local high_memory_pods=()

    while IFS= read -r line; do
        local pod_name=$(echo "$line" | awk '{print $1}')
        local cpu_usage=$(echo "$line" | awk '{print $2}' | sed 's/m//')
        local memory_usage=$(echo "$line" | awk '{print $3}' | sed 's/Mi//')

        # Check for high CPU usage (>1000m = 1 CPU)
        if [[ "$cpu_usage" =~ ^[0-9]+$ ]] && [[ "$cpu_usage" -gt 2000 ]]; then
            high_cpu_pods+=("$pod_name:${cpu_usage}m")
        fi

        # Check for high memory usage (>2Gi = 2048Mi)
        if [[ "$memory_usage" =~ ^[0-9]+$ ]] && [[ "$memory_usage" -gt 3072 ]]; then
            high_memory_pods+=("$pod_name:${memory_usage}Mi")
        fi

    done < <(kubectl top pods -n "$NAMESPACE" --no-headers 2>/dev/null || echo "")

    if [[ ${#high_cpu_pods[@]} -gt 0 ]]; then
        log_warning "High CPU usage pods: ${high_cpu_pods[*]}"
        send_alert "WARNING" "High CPU usage pods: ${high_cpu_pods[*]}"
    fi

    if [[ ${#high_memory_pods[@]} -gt 0 ]]; then
        log_warning "High memory usage pods: ${high_memory_pods[*]}"
        send_alert "WARNING" "High memory usage pods: ${high_memory_pods[*]}"
    fi

    log_success "Resource usage check completed"
    return 0
}

# Check persistent volumes
check_persistent_volumes() {
    log_info "Checking persistent volumes..."

    local failed_pvs=()

    while IFS= read -r line; do
        local pv_name=$(echo "$line" | awk '{print $1}')
        local status=$(echo "$line" | awk '{print $2}')

        if [[ "$status" != "Bound" && "$status" != "Available" ]]; then
            failed_pvs+=("$pv_name:$status")
        fi

    done < <(kubectl get pv --no-headers 2>/dev/null || echo "")

    if [[ ${#failed_pvs[@]} -gt 0 ]]; then
        log_warning "PV issues detected: ${failed_pvs[*]}"
        send_alert "WARNING" "PV issues: ${failed_pvs[*]}"
    fi

    # Check PVC status
    local failed_pvcs=()

    while IFS= read -r line; do
        local pvc_name=$(echo "$line" | awk '{print $1}')
        local status=$(echo "$line" | awk '{print $2}')

        if [[ "$status" != "Bound" ]]; then
            failed_pvcs+=("$pvc_name:$status")
        fi

    done < <(kubectl get pvc -n "$NAMESPACE" --no-headers 2>/dev/null || echo "")

    if [[ ${#failed_pvcs[@]} -gt 0 ]]; then
        log_error "PVC issues detected: ${failed_pvcs[*]}"
        send_alert "CRITICAL" "PVC issues: ${failed_pvcs[*]}"
        return 1
    fi

    log_success "Persistent volumes check completed"
    return 0
}

# Generate system report
generate_system_report() {
    local report_file="/tmp/freedom-stock-system-report-$(date +%Y%m%d-%H%M%S).txt"

    {
        echo "Freedom Stock Analysis System Health Report"
        echo "=========================================="
        echo "Generated: $(date)"
        echo "Namespace: $NAMESPACE"
        echo ""

        echo "DEPLOYMENTS:"
        kubectl get deployments -n "$NAMESPACE" -o wide 2>/dev/null || echo "Unable to get deployments"
        echo ""

        echo "PODS:"
        kubectl get pods -n "$NAMESPACE" -o wide 2>/dev/null || echo "Unable to get pods"
        echo ""

        echo "SERVICES:"
        kubectl get services -n "$NAMESPACE" -o wide 2>/dev/null || echo "Unable to get services"
        echo ""

        echo "PERSISTENT VOLUMES:"
        kubectl get pv 2>/dev/null || echo "Unable to get PVs"
        echo ""

        echo "PERSISTENT VOLUME CLAIMS:"
        kubectl get pvc -n "$NAMESPACE" 2>/dev/null || echo "Unable to get PVCs"
        echo ""

        echo "NODE RESOURCE USAGE:"
        kubectl top nodes 2>/dev/null || echo "Unable to get node usage"
        echo ""

        echo "POD RESOURCE USAGE:"
        kubectl top pods -n "$NAMESPACE" 2>/dev/null || echo "Unable to get pod usage"
        echo ""

        echo "EVENTS (last 1 hour):"
        kubectl get events -n "$NAMESPACE" --sort-by=.metadata.creationTimestamp | tail -20 2>/dev/null || echo "Unable to get events"

    } > "$report_file"

    log_info "System report generated: $report_file"
    echo "$report_file"
}

# Main monitoring function
main() {
    local exit_code=0

    log_info "Starting Freedom Stock Analysis System health check..."

    # Create log directory if it doesn't exist
    mkdir -p "$(dirname "$LOG_FILE")"

    # Run all checks
    check_cluster_connectivity || exit_code=1
    check_pods_status || exit_code=1
    check_deployments_status || exit_code=1
    check_service_endpoints || exit_code=1
    check_database_connectivity || exit_code=1
    check_resource_usage || exit_code=1
    check_persistent_volumes || exit_code=1

    # Generate report
    local report_file
    report_file=$(generate_system_report)

    if [[ $exit_code -eq 0 ]]; then
        log_success "All health checks passed"
        send_alert "INFO" "System health check completed successfully"
    else
        log_error "Some health checks failed"
        send_alert "CRITICAL" "System health check failed. Report: $report_file"
    fi

    exit $exit_code
}

# Handle command line arguments
case "${1:-monitor}" in
    monitor)
        main
        ;;
    report)
        generate_system_report
        ;;
    *)
        echo "Usage: $0 [monitor|report]"
        echo "  monitor  - Run health checks (default)"
        echo "  report   - Generate system report only"
        exit 1
        ;;
esac