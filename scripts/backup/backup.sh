#!/bin/bash
# Backup Script for Freedom Stock Analysis System
# Version: 1.0

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NAMESPACE="${NAMESPACE:-freedom-stock-prod}"
BACKUP_ROOT="${BACKUP_ROOT:-/backup/freedom-stock}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
S3_BUCKET="${S3_BUCKET:-}"
S3_PREFIX="${S3_PREFIX:-freedom-stock-backups}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Create backup directory structure
create_backup_structure() {
    local backup_date=$(date +%Y%m%d-%H%M%S)
    local backup_dir="$BACKUP_ROOT/$backup_date"

    mkdir -p "$backup_dir"/{kubernetes,database,configs,logs}
    echo "$backup_dir"
}

# Backup Kubernetes resources
backup_kubernetes_resources() {
    local backup_dir="$1"
    local k8s_dir="$backup_dir/kubernetes"

    log_info "Backing up Kubernetes resources..."

    # Backup all resources in the namespace
    local resources=("deployments" "services" "configmaps" "secrets" "persistentvolumeclaims" "ingresses" "networkpolicies")

    for resource in "${resources[@]}"; do
        log_info "Backing up $resource..."
        kubectl get "$resource" -n "$NAMESPACE" -o yaml > "$k8s_dir/$resource.yaml" 2>/dev/null || {
            log_warning "Failed to backup $resource"
        }
    done

    # Backup cluster-wide resources related to the application
    kubectl get persistentvolumes -o yaml > "$k8s_dir/persistentvolumes.yaml" 2>/dev/null || true
    kubectl get storageclasses -o yaml > "$k8s_dir/storageclasses.yaml" 2>/dev/null || true

    # Save current image versions
    log_info "Saving current image versions..."
    kubectl get deployments -n "$NAMESPACE" -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.template.spec.containers[*].image}{"\n"}{end}' > "$k8s_dir/current-images.txt"

    # Save resource usage
    kubectl top pods -n "$NAMESPACE" > "$k8s_dir/pod-resource-usage.txt" 2>/dev/null || echo "Resource metrics not available" > "$k8s_dir/pod-resource-usage.txt"
    kubectl top nodes > "$k8s_dir/node-resource-usage.txt" 2>/dev/null || echo "Node metrics not available" > "$k8s_dir/node-resource-usage.txt"

    log_success "Kubernetes resources backed up"
}

# Backup ClickHouse database
backup_clickhouse_database() {
    local backup_dir="$1"
    local db_dir="$backup_dir/database"

    log_info "Backing up ClickHouse database..."

    # Get ClickHouse pod
    local clickhouse_pod=$(kubectl get pod -l app=clickhouse-primary -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [[ -z "$clickhouse_pod" ]]; then
        log_error "ClickHouse pod not found"
        return 1
    fi

    # Create database backup
    log_info "Creating ClickHouse schema backup..."
    kubectl exec -n "$NAMESPACE" "$clickhouse_pod" -- clickhouse-client --query "SHOW CREATE DATABASE freedom" > "$db_dir/schema.sql" 2>/dev/null || {
        log_warning "Could not backup database schema"
    }

    # Get list of tables
    local tables=$(kubectl exec -n "$NAMESPACE" "$clickhouse_pod" -- clickhouse-client --query "SHOW TABLES FROM freedom" 2>/dev/null || echo "")

    if [[ -n "$tables" ]]; then
        echo "$tables" > "$db_dir/tables.list"

        # Backup table schemas
        log_info "Backing up table schemas..."
        while IFS= read -r table; do
            if [[ -n "$table" ]]; then
                log_info "Backing up schema for table: $table"
                kubectl exec -n "$NAMESPACE" "$clickhouse_pod" -- clickhouse-client --query "SHOW CREATE TABLE freedom.$table" > "$db_dir/schema_$table.sql" 2>/dev/null || {
                    log_warning "Could not backup schema for table $table"
                }
            fi
        done <<< "$tables"

        # Backup critical tables data (sample data for recovery testing)
        log_info "Backing up sample data..."
        while IFS= read -r table; do
            if [[ -n "$table" ]]; then
                log_info "Backing up sample data for table: $table"
                kubectl exec -n "$NAMESPACE" "$clickhouse_pod" -- clickhouse-client --query "SELECT * FROM freedom.$table LIMIT 1000 FORMAT TSV" > "$db_dir/sample_data_$table.tsv" 2>/dev/null || {
                    log_warning "Could not backup sample data for table $table"
                }
            fi
        done <<< "$tables"
    else
        log_warning "No tables found in database"
    fi

    # Backup ClickHouse configuration
    log_info "Backing up ClickHouse configuration..."
    kubectl exec -n "$NAMESPACE" "$clickhouse_pod" -- cat /etc/clickhouse-server/config.xml > "$db_dir/config.xml" 2>/dev/null || {
        log_warning "Could not backup ClickHouse config"
    }

    kubectl exec -n "$NAMESPACE" "$clickhouse_pod" -- cat /etc/clickhouse-server/users.xml > "$db_dir/users.xml" 2>/dev/null || {
        log_warning "Could not backup ClickHouse users config"
    }

    log_success "ClickHouse database backed up"
}

# Backup Redis data
backup_redis_data() {
    local backup_dir="$1"
    local redis_dir="$backup_dir/database"

    log_info "Backing up Redis data..."

    # Get Redis pod
    local redis_pod=$(kubectl get pod -l app=redis -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [[ -z "$redis_pod" ]]; then
        log_warning "Redis pod not found, skipping Redis backup"
        return 0
    fi

    # Create Redis backup
    kubectl exec -n "$NAMESPACE" "$redis_pod" -- redis-cli BGSAVE > /dev/null 2>&1 || {
        log_warning "Could not trigger Redis background save"
        return 0
    }

    # Wait for backup to complete
    sleep 5

    # Copy the dump file
    kubectl cp -n "$NAMESPACE" "$redis_pod":/data/dump.rdb "$redis_dir/redis-dump.rdb" 2>/dev/null || {
        log_warning "Could not copy Redis dump file"
    }

    # Get Redis info
    kubectl exec -n "$NAMESPACE" "$redis_pod" -- redis-cli INFO > "$redis_dir/redis-info.txt" 2>/dev/null || {
        log_warning "Could not get Redis info"
    }

    log_success "Redis data backed up"
}

# Backup application configurations
backup_application_configs() {
    local backup_dir="$1"
    local config_dir="$backup_dir/configs"

    log_info "Backing up application configurations..."

    # Copy local configuration files
    if [[ -d "$PROJECT_ROOT/config" ]]; then
        cp -r "$PROJECT_ROOT/config" "$config_dir/app-config"
    fi

    if [[ -d "$PROJECT_ROOT/k8s" ]]; then
        cp -r "$PROJECT_ROOT/k8s" "$config_dir/k8s-manifests"
    fi

    # Backup Docker configurations
    if [[ -f "$PROJECT_ROOT/docker-compose.prod.yml" ]]; then
        cp "$PROJECT_ROOT/docker-compose.prod.yml" "$config_dir/"
    fi

    if [[ -f "$PROJECT_ROOT/Dockerfile" ]]; then
        cp "$PROJECT_ROOT/Dockerfile" "$config_dir/"
    fi

    if [[ -f "$PROJECT_ROOT/Dockerfile.processor" ]]; then
        cp "$PROJECT_ROOT/Dockerfile.processor" "$config_dir/"
    fi

    # Backup CI/CD configurations
    if [[ -f "$PROJECT_ROOT/.github/workflows/ci-cd.yml" ]]; then
        mkdir -p "$config_dir/.github/workflows"
        cp "$PROJECT_ROOT/.github/workflows/ci-cd.yml" "$config_dir/.github/workflows/"
    fi

    if [[ -f "$PROJECT_ROOT/.gitlab-ci.yml" ]]; then
        cp "$PROJECT_ROOT/.gitlab-ci.yml" "$config_dir/"
    fi

    log_success "Application configurations backed up"
}

# Backup application logs
backup_application_logs() {
    local backup_dir="$1"
    local logs_dir="$backup_dir/logs"

    log_info "Backing up application logs..."

    # Get recent logs from pods
    local pods=$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)

    for pod in $pods; do
        if [[ -n "$pod" ]]; then
            log_info "Backing up logs for pod: $pod"
            kubectl logs "$pod" -n "$NAMESPACE" --tail=1000 > "$logs_dir/$pod.log" 2>/dev/null || {
                log_warning "Could not get logs for pod $pod"
            }

            # Get previous logs if available
            kubectl logs "$pod" -n "$NAMESPACE" --previous --tail=1000 > "$logs_dir/$pod-previous.log" 2>/dev/null || true
        fi
    done

    # Get events
    kubectl get events -n "$NAMESPACE" --sort-by=.metadata.creationTimestamp > "$logs_dir/kubernetes-events.log" 2>/dev/null || {
        log_warning "Could not get Kubernetes events"
    }

    log_success "Application logs backed up"
}

# Create backup manifest
create_backup_manifest() {
    local backup_dir="$1"
    local manifest_file="$backup_dir/backup-manifest.json"

    log_info "Creating backup manifest..."

    cat > "$manifest_file" << EOF
{
  "backup_info": {
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "namespace": "$NAMESPACE",
    "backup_version": "1.0",
    "backup_type": "full"
  },
  "system_info": {
    "kubernetes_version": "$(kubectl version --client --short 2>/dev/null | grep 'Client Version' || echo 'unknown')",
    "cluster_info": "$(kubectl cluster-info 2>/dev/null | head -1 || echo 'unknown')"
  },
  "components": {
    "kubernetes_resources": "kubernetes/",
    "clickhouse_database": "database/",
    "redis_data": "database/",
    "application_configs": "configs/",
    "application_logs": "logs/"
  },
  "files": [
$(find "$backup_dir" -type f -printf '    "%P",\n' | sed '$s/,$//')
  ]
}
EOF

    log_success "Backup manifest created"
}

# Compress backup
compress_backup() {
    local backup_dir="$1"
    local backup_name=$(basename "$backup_dir")
    local compressed_file="$backup_dir.tar.gz"

    log_info "Compressing backup..."

    cd "$(dirname "$backup_dir")"
    tar -czf "$compressed_file" "$backup_name"

    # Remove uncompressed directory
    rm -rf "$backup_dir"

    log_success "Backup compressed: $compressed_file"
    echo "$compressed_file"
}

# Upload to S3 (if configured)
upload_to_s3() {
    local backup_file="$1"

    if [[ -z "$S3_BUCKET" ]]; then
        log_info "S3 upload not configured, skipping"
        return 0
    fi

    log_info "Uploading backup to S3..."

    local s3_key="$S3_PREFIX/$(basename "$backup_file")"

    if command -v aws &> /dev/null; then
        aws s3 cp "$backup_file" "s3://$S3_BUCKET/$s3_key" || {
            log_error "Failed to upload to S3"
            return 1
        }
        log_success "Backup uploaded to S3: s3://$S3_BUCKET/$s3_key"
    else
        log_warning "AWS CLI not found, cannot upload to S3"
        return 1
    fi
}

# Clean old backups
cleanup_old_backups() {
    log_info "Cleaning up old backups (older than $RETENTION_DAYS days)..."

    find "$BACKUP_ROOT" -name "*.tar.gz" -type f -mtime +$RETENTION_DAYS -delete 2>/dev/null || {
        log_warning "Could not clean old local backups"
    }

    # Clean S3 backups if configured
    if [[ -n "$S3_BUCKET" ]] && command -v aws &> /dev/null; then
        local cutoff_date=$(date -d "$RETENTION_DAYS days ago" +%Y%m%d)

        aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/" | while read -r line; do
            local file_date=$(echo "$line" | awk '{print $1}' | tr -d '-')
            local file_name=$(echo "$line" | awk '{print $4}')

            if [[ "$file_date" < "$cutoff_date" ]]; then
                aws s3 rm "s3://$S3_BUCKET/$S3_PREFIX/$file_name" || {
                    log_warning "Could not delete old S3 backup: $file_name"
                }
            fi
        done
    fi

    log_success "Old backups cleaned up"
}

# Restore function (basic)
restore_backup() {
    local backup_file="$1"
    local restore_dir="/tmp/freedom-stock-restore-$(date +%Y%m%d-%H%M%S)"

    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi

    log_info "Extracting backup to: $restore_dir"
    mkdir -p "$restore_dir"
    tar -xzf "$backup_file" -C "$restore_dir" --strip-components=1

    log_info "Backup extracted. Manual restoration steps required:"
    echo "1. Review Kubernetes manifests in: $restore_dir/kubernetes/"
    echo "2. Review database backups in: $restore_dir/database/"
    echo "3. Review configurations in: $restore_dir/configs/"
    echo "4. Apply Kubernetes resources as needed"
    echo "5. Restore database data using appropriate tools"

    return 0
}

# Main function
main() {
    local action="${1:-backup}"

    case "$action" in
        backup)
            log_info "Starting Freedom Stock Analysis System backup..."

            # Create backup directory
            local backup_dir
            backup_dir=$(create_backup_structure)

            log_info "Backup directory: $backup_dir"

            # Perform backup
            backup_kubernetes_resources "$backup_dir"
            backup_clickhouse_database "$backup_dir"
            backup_redis_data "$backup_dir"
            backup_application_configs "$backup_dir"
            backup_application_logs "$backup_dir"
            create_backup_manifest "$backup_dir"

            # Compress backup
            local compressed_backup
            compressed_backup=$(compress_backup "$backup_dir")

            # Upload to S3 if configured
            upload_to_s3 "$compressed_backup"

            # Cleanup old backups
            cleanup_old_backups

            log_success "Backup completed: $compressed_backup"
            ;;

        restore)
            local backup_file="$2"
            if [[ -z "$backup_file" ]]; then
                log_error "Backup file required for restore"
                echo "Usage: $0 restore <backup-file>"
                exit 1
            fi
            restore_backup "$backup_file"
            ;;

        *)
            echo "Usage: $0 [backup|restore] [backup-file]"
            echo "  backup          - Create backup (default)"
            echo "  restore <file>  - Restore from backup file"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"