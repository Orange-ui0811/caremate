-- 会话软删除：保留工单与审核来源，只移除客户侧会话入口。
ALTER TABLE conversations ADD COLUMN deleted_at DATETIME NULL;
