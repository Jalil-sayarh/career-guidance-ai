/*! START TRANSACTION */;
CREATE TABLE riasec_keywords (
  element_id CHARACTER VARYING(20) NOT NULL,
  keyword CHARACTER VARYING(150) NOT NULL,
  keyword_type CHARACTER VARYING(20) NOT NULL,
  FOREIGN KEY (element_id) REFERENCES content_model_reference(element_id));
/*! COMMIT */;
/*! START TRANSACTION */;

INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.a', 'Build', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.a', 'Drive', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.a', 'Install', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.a', 'Maintain', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.a', 'Repair', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.a', 'Work with Hands', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.a', 'Animals', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.a', 'Electronics', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.a', 'Equipment', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.a', 'Machines', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.a', 'Plants', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.a', 'Real-World Materials like Wood', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.a', 'Tools', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.b', 'Analyze', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.b', 'Diagnose', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.b', 'Discover', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.b', 'Problem Solve', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.b', 'Research', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.b', 'Study', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.b', 'Test', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.b', 'Think', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.b', 'Facts', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.b', 'Ideas', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.b', 'Knowledge', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.b', 'Laboratory', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.b', 'Science', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.c', 'Compose', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.c', 'Create', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.c', 'Dance', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.c', 'Design', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.c', 'Perform', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.c', 'Self-Express', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.c', 'Write', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.c', 'Art', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.c', 'Graphics', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.c', 'Media', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.c', 'Music', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.c', 'Theatre', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.d', 'Advise', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.d', 'Educate', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.d', 'Guide', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.d', 'Help', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.d', 'Nurture', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.d', 'Teach', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.d', 'Communication', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.d', 'Health', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.d', 'People', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.d', 'Service', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.d', 'Social', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.e', 'Direct', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.e', 'Lead', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.e', 'Manage', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.e', 'Market', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.e', 'Negotiate', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.e', 'Sell', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.e', 'Supervise', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.e', 'Business', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.e', 'Customer', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.e', 'Employee', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.e', 'Law', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.e', 'Politics', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.e', 'Product', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.f', 'Attention to Detail', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.f', 'File', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.f', 'Inspect', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.f', 'Organize', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.f', 'Record', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.f', 'Sort', 'Action');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.f', 'Data', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.f', 'Files', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.f', 'Information', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.f', 'Office', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.f', 'Procedures', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.f', 'Regulations', 'Object');
INSERT INTO riasec_keywords (element_id, keyword, keyword_type) VALUES ('1.B.1.f', 'Rules', 'Object');
/*! COMMIT */;

