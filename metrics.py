def write_health_status(self, health_status: dict) -> None:
    try:
        with open(self.health_file, 'w') as f:
            json.dump(health_status, f)
    except OSError as e:
        self.logger.error(f'Failed to write health status: {e}
')
        # Best-effort write, continue running
