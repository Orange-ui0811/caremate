-- Issue #3 workflow events, plans and local ticket receipts.
CREATE TABLE IF NOT EXISTS workflow_records (
    `key` VARCHAR(192) PRIMARY KEY,
    conversation_id BIGINT NOT NULL,
    payload JSON NOT NULL,
    INDEX ix_workflow_records_conversation_id (conversation_id)
);
-- Existing installations use workflow.ensure_tables(), preserving other enum values.
