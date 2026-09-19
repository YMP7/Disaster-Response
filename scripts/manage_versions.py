"""Command-line Model Version Management & Rollback Utility.
Disaster Response AI Platform - SemVer 2.0.0

Provides command center operators with instant inspection, metric comparison,
explicit activation, zero-downtime rollback, and SHA-256 checksum verification.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Ensure repo root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.version import __version__
from models.registry import ModelRegistry
from orchestration.audit_log import CryptographicAuditLog

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    CONSOLE = Console()
    HAS_RICH = True
except ImportError:
    CONSOLE = None
    HAS_RICH = False


def _format_metrics(metrics: dict) -> str:
    """Format dictionary of metrics into a compact, readable string."""
    parts = []
    if "val_accuracy" in metrics:
        parts.append(f"acc={metrics['val_accuracy']*100:.2f}%")
    if "macro_f1" in metrics:
        parts.append(f"F1={metrics['macro_f1']:.4f}")
    if "val_damage_accuracy" in metrics:
        parts.append(f"dmg_acc={metrics['val_damage_accuracy']*100:.1f}%")
    if "val_ordinal_mae" in metrics:
        parts.append(f"MAE={metrics['val_ordinal_mae']:.4f}")
    if "val_road_passability_accuracy" in metrics:
        parts.append(f"road_acc={metrics['val_road_passability_accuracy']*100:.2f}%")
    if "water_mask_mean_iou" in metrics:
        parts.append(f"mIoU={metrics['water_mask_mean_iou']:.4f}")
    if "water_extent_mae_pct" in metrics:
        parts.append(f"ext_mae={metrics['water_extent_mae_pct']:.2f}%")
    if not parts and metrics:
        first_few = list(metrics.items())[:2]
        parts = [f"{k}={v}" for k, v in first_few]
    return ", ".join(parts) if parts else "N/A"


def cmd_list(registry: ModelRegistry, stage: Optional[str] = None):
    """Lists registered model versions."""
    versions = registry.list_versions(stage=stage)
    if not versions:
        print(f"No registered models found" + (f" for stage '{stage}'" if stage else "") + ".")
        return

    if HAS_RICH:
        title = f"Disaster Response Model Registry [bold cyan](Platform v{__version__})[/bold cyan]"
        if stage:
            title += f" - Stage: [bold yellow]{stage}[/bold yellow]"
        table = Table(title=title, box=box.ROUNDED, header_style="bold magenta")
        table.add_column("Stage", style="cyan", no_wrap=True)
        table.add_column("Model ID", style="bold white")
        table.add_column("Version", style="magenta")
        table.add_column("Active", justify="center")
        table.add_column("Status", justify="center")
        table.add_column("Primary Metrics", style="green")
        table.add_column("SHA-256 (Prefix)", style="dim")

        for item in versions:
            active_str = "[bold green]ACTIVE[/bold green]" if item["is_active"] else "[dim]INACTIVE[/dim]"
            status_str = (
                "[green]active[/green]" if item["status"] == "active"
                else "[yellow]ineffective[/yellow]" if "ineffective" in item["status"]
                else "[red]inadequate[/red]"
            )
            sha_short = item["sha256_checksum"][:10] + "..." if len(item["sha256_checksum"]) > 10 else item["sha256_checksum"]
            table.add_row(
                item["stage"],
                item["model_id"],
                item["version"],
                active_str,
                status_str,
                _format_metrics(item.get("metrics", {})),
                sha_short
            )
        CONSOLE.print(table)
    else:
        print(f"=== Registered Models (Platform v{__version__}) ===")
        for item in versions:
            flag = "[ACTIVE]" if item["is_active"] else "[ ]"
            print(f"{flag} {item['stage']:<26} {item['model_id']:<32} v{item['version']:<8} status={item['status']:<16} metrics: {_format_metrics(item.get('metrics', {}))}")


def cmd_compare(registry: ModelRegistry, stage: str):
    """Displays comparative view of all registered versions for a given stage."""
    comparison = registry.get_version_comparison(stage=stage)
    versions = comparison.get("versions", [])
    if not versions:
        print(f"No models registered for stage '{stage}'.")
        return

    if HAS_RICH:
        table = Table(
            title=f"Version Comparison: Stage [bold yellow]{stage}[/bold yellow] (Active: [bold green]{comparison.get('active_version')}[/bold green])",
            box=box.HEAVY_EDGE,
            header_style="bold cyan"
        )
        table.add_column("Version", style="bold magenta", justify="center")
        table.add_column("Model ID", style="bold white")
        table.add_column("Architecture", style="italic")
        table.add_column("Active", justify="center")
        table.add_column("Status", justify="center")
        table.add_column("Key Metrics", style="green")
        table.add_column("Notes", style="dim", max_width=40)

        for v in versions:
            active_str = "[bold green]YES (Active)[/bold green]" if v["is_active"] else "[dim]No[/dim]"
            status_str = (
                "[green]active[/green]" if v["status"] == "active"
                else "[yellow]ineffective[/yellow]" if "ineffective" in v["status"]
                else "[red]inadequate[/red]"
            )
            table.add_row(
                v["version"],
                v["model_id"],
                v["architecture"],
                active_str,
                status_str,
                _format_metrics(v.get("metrics", {})),
                v.get("notes") or "-"
            )
        CONSOLE.print(table)
    else:
        print(f"=== Version Comparison for Stage '{stage}' ===")
        print(f"Active: {comparison.get('active_model_id')} (v{comparison.get('active_version')})")
        for v in versions:
            flag = "[ACTIVE]" if v["is_active"] else "[      ]"
            print(f"  {flag} v{v['version']:<6} | {v['model_id']:<32} | {_format_metrics(v.get('metrics', {}))}")


def cmd_activate(registry: ModelRegistry, model_id: str, operator: str = "OPS_CONSOLE", reason: str = "CLI Operator Activation"):
    """Activates a model artifact by its model_id."""
    try:
        activated = registry.activate_model(model_id)
        # Log to cryptographic audit log
        audit = CryptographicAuditLog()
        audit.append_block(
            event_type="MODEL_ACTIVATION",
            actor=operator,
            details={
                "model_id": activated.model_id,
                "version": activated.version,
                "stage": activated.stage,
                "sha256": activated.sha256_checksum,
                "reason": reason
            }
        )
        msg = f"Successfully activated [bold white]{activated.model_id}[/bold white] (v{activated.version}) for stage [bold yellow]{activated.stage}[/bold yellow]."
        if HAS_RICH:
            CONSOLE.print(Panel(msg, title="Model Activated", style="bold green", box=box.ROUNDED))
        else:
            print(f"SUCCESS: {msg}")
    except KeyError as e:
        if HAS_RICH:
            CONSOLE.print(f"[bold red]Error:[/bold red] {e}")
        else:
            print(f"ERROR: {e}")
        sys.exit(1)


def cmd_rollback(registry: ModelRegistry, stage: str, target: str, operator: str = "OPS_CONSOLE", reason: str = "CLI Operator Rollback"):
    """Rolls back the active model for a stage to a designated target version or model_id."""
    prev_active = registry.get_active_model(stage)
    rolled_back = registry.rollback(stage=stage, target=target)
    if not rolled_back:
        err = f"No matching model found for stage '{stage}' with target '{target}'."
        if HAS_RICH:
            CONSOLE.print(f"[bold red]Rollback Failed:[/bold red] {err}")
        else:
            print(f"ERROR: {err}")
        sys.exit(1)

    # Log to cryptographic audit log
    audit = CryptographicAuditLog()
    audit.append_block(
        event_type="MODEL_ROLLBACK",
        actor=operator,
        details={
            "stage": stage,
            "target": target,
            "previous_model_id": prev_active.model_id if prev_active else None,
            "previous_version": prev_active.version if prev_active else None,
            "new_model_id": rolled_back.model_id,
            "new_version": rolled_back.version,
            "new_sha256": rolled_back.sha256_checksum,
            "reason": reason
        }
    )

    msg = (
        f"Stage [bold yellow]{stage}[/bold yellow] rolled back to:\n"
        f"  Model ID: [bold white]{rolled_back.model_id}[/bold white]\n"
        f"  Version:  [bold magenta]v{rolled_back.version}[/bold magenta]\n"
        f"  Checksum: [dim]{rolled_back.sha256_checksum[:16]}...[/dim]\n"
        f"  Previous: {prev_active.model_id if prev_active else 'None'}"
    )
    if HAS_RICH:
        CONSOLE.print(Panel(msg, title="Zero-Downtime Rollback Complete", style="bold yellow", box=box.ROUNDED))
    else:
        print(f"SUCCESS: Rollback complete for stage {stage} to {rolled_back.model_id} (v{rolled_back.version})")


def cmd_verify(registry: ModelRegistry, stage: Optional[str] = None):
    """Verifies that weights files exist on disk and match registered SHA-256 checksums."""
    results = registry.verify_checksums(stage=stage)
    all_valid = all(v.get("valid", False) for v in results.values())

    if HAS_RICH:
        table = Table(
            title=f"Cryptographic Integrity & File Presence Check (Stage: {stage or 'ALL'})",
            box=box.ROUNDED,
            header_style="bold blue"
        )
        table.add_column("Model ID", style="bold white")
        table.add_column("Status", justify="center")
        table.add_column("File Path", style="dim")
        table.add_column("Checksum Match", justify="center")

        for mid, r in results.items():
            if r["status"] == "verified":
                status_cell = "[bold green]VERIFIED[/bold green]"
                match_cell = "[green]MATCH[/green]"
            elif r["status"] == "skipped_rule_based":
                status_cell = "[cyan]HEURISTIC (Rule-based)[/cyan]"
                match_cell = "[cyan]N/A[/cyan]"
            elif r["status"] == "missing_file":
                status_cell = "[bold red]FILE MISSING[/bold red]"
                match_cell = "[red]NO FILE[/red]"
            else:
                status_cell = "[bold red]MISMATCH[/bold red]"
                match_cell = "[red]FAIL[/red]"
            table.add_row(mid, status_cell, r.get("path", "none"), match_cell)
        CONSOLE.print(table)
        if all_valid:
            CONSOLE.print("[bold green][OK] All registered model weights verified with SHA-256 cryptographic match.[/bold green]")
        else:
            CONSOLE.print("[bold red][FAIL] One or more model weight files are missing or corrupted.[/bold red]")
    else:
        print("=== Model Weights Integrity Verification ===")
        for mid, r in results.items():
            print(f"  {mid:<32} status={r['status']:<16} valid={r.get('valid')}")


def main():
    parser = argparse.ArgumentParser(
        description=f"India Disaster Response AI Platform - Model Version Manager (v{__version__})"
    )
    subparsers = parser.add_subparsers(dest="command", help="Management command to execute")

    # list
    p_list = subparsers.add_parser("list", help="List registered model versions")
    p_list.add_argument("--stage", type=str, default=None, help="Filter by pipeline stage")

    # compare
    p_comp = subparsers.add_parser("compare", help="Compare versions for a specific pipeline stage")
    p_comp.add_argument("--stage", type=str, required=True, help="Stage to compare (e.g. stage1_triage, stage2_severity)")

    # activate
    p_act = subparsers.add_parser("activate", help="Explicitly activate a specific model ID")
    p_act.add_argument("--model-id", type=str, required=True, help="Model ID to activate")
    p_act.add_argument("--operator", type=str, default="OPS_CONSOLE", help="Operator identifier for audit ledger")
    p_act.add_argument("--reason", type=str, default="CLI manual activation", help="Justification for audit log")

    # rollback
    p_rb = subparsers.add_parser("rollback", help="Roll back active model of a stage to prior version/ID")
    p_rb.add_argument("--stage", type=str, required=True, help="Stage to rollback")
    p_rb.add_argument("--target", type=str, required=True, help="Target model ID, version string, or prefix (e.g. '1.1.0' or 'v1')")
    p_rb.add_argument("--operator", type=str, default="OPS_CONSOLE", help="Operator identifier for audit ledger")
    p_rb.add_argument("--reason", type=str, default="CLI operator rollback", help="Justification for audit log")

    # verify
    p_ver = subparsers.add_parser("verify", help="Verify SHA-256 checksums and file presence on disk")
    p_ver.add_argument("--stage", type=str, default=None, help="Filter verification to a specific stage")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    registry = ModelRegistry()

    if args.command == "list":
        cmd_list(registry, stage=args.stage)
    elif args.command == "compare":
        cmd_compare(registry, stage=args.stage)
    elif args.command == "activate":
        cmd_activate(registry, model_id=args.model_id, operator=args.operator, reason=args.reason)
    elif args.command == "rollback":
        cmd_rollback(registry, stage=args.stage, target=args.target, operator=args.operator, reason=args.reason)
    elif args.command == "verify":
        cmd_verify(registry, stage=args.stage)


if __name__ == "__main__":
    main()
