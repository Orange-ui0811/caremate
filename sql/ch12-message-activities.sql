-- 保存回答对应的检索及工具过程，历史回载时显示。
ALTER TABLE messages ADD COLUMN activities JSON NULL;
