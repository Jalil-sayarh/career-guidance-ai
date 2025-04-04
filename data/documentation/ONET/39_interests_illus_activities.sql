/*! START TRANSACTION */;
CREATE TABLE interests_illus_activities (
  element_id CHARACTER VARYING(20) NOT NULL,
  interest_type CHARACTER VARYING(20) NOT NULL,
  activity CHARACTER VARYING(150) NOT NULL,
  FOREIGN KEY (element_id) REFERENCES content_model_reference(element_id));
/*! COMMIT */;
/*! START TRANSACTION */;

INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.1.a', 'General', 'Build kitchen cabinets.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.1.a', 'General', 'Drive a truck to deliver packages to offices and homes.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.1.a', 'General', 'Put out forest fires.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.1.a', 'General', 'Repair household appliances.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.1.b', 'General', 'Develop a new medicine.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.1.b', 'General', 'Develop a way to better predict the weather.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.1.b', 'General', 'Study ways to reduce water pollution.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.1.b', 'General', 'Work in a biology lab.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.1.c', 'General', 'Compose or arrange music.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.1.c', 'General', 'Create special effects for movies.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.1.c', 'General', 'Draw pictures.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.1.c', 'General', 'Write books or plays.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.1.d', 'General', 'Give career guidance to people.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.1.d', 'General', 'Help people with personal or emotional problems.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.1.d', 'General', 'Perform rehabilitation therapy.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.1.d', 'General', 'Teach a high school class.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.1.e', 'General', 'Manage a department within a large company.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.1.e', 'General', 'Market a new line of clothing.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.1.e', 'General', 'Represent a client in a lawsuit.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.1.e', 'General', 'Start your own business.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.1.f', 'General', 'Develop a spreadsheet using computer software.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.1.f', 'General', 'Inventory supplies using a handheld computer.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.1.f', 'General', 'Keep shipping and receiving records.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.1.f', 'General', 'Load computer software into a large computer network.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.a', 'Basic', 'Install computer systems.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.a', 'Basic', 'Maintain aircraft engines.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.a', 'Basic', 'Make repairs to industrial robots.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.a', 'Basic', 'Repair motors and generators.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.b', 'Basic', 'Build and finish furniture.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.b', 'Basic', 'Build and install cabinets.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.b', 'Basic', 'Construct wood stairways.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.b', 'Basic', 'Operate woodworking machines.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.c', 'Basic', 'Drive a bus or motor coach.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.c', 'Basic', 'Operate a bulldozer to move dirt.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.c', 'Basic', 'Operate a forklift to move boxes.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.c', 'Basic', 'Transport materials in a truck or van.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.d', 'Basic', 'Clean a work area.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.d', 'Basic', 'Manually move cargo and freight.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.d', 'Basic', 'Perform physical labor at a construction site.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.d', 'Basic', 'Unload, sort, and move materials, equipment, or supplies.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.e', 'Basic', 'Apprehend and arrest suspects.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.e', 'Basic', 'Guard inmates in a prison.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.e', 'Basic', 'Investigate criminal activity.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.e', 'Basic', 'Maintain order and protect life by enforcing laws.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.f', 'Basic', 'Develop plans to reduce soil erosion and protect rangelands.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.f', 'Basic', 'Inspect crops for disease.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.f', 'Basic', 'Plant, cultivate, and harvest crops.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.f', 'Basic', 'Raise and tend to farm animals.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.g', 'Basic', 'Conserve wildlife habitats.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.g', 'Basic', 'Manage public forested lands for recreational purposes.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.g', 'Basic', 'Plant a public garden.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.g', 'Basic', 'Train conservation workers in planting tree seedlings.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.h', 'Basic', 'Exercise animals in a shelter.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.h', 'Basic', 'Provide treatment to sick or injured animals.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.h', 'Basic', 'Supply animals with food, water, and personal care.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.h', 'Basic', 'Train animals to assist persons with disabilities.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.i', 'Basic', 'Coach a sports team.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.i', 'Basic', 'Compete in an athletic event.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.i', 'Basic', 'Play sports.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.i', 'Basic', 'Train for an athletic event.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.j', 'Basic', 'Apply principles of engineering in designing buildings.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.j', 'Basic', 'Design, construct, and test aircraft, missiles, and spacecraft.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.j', 'Basic', 'Explain engineering drawings.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.j', 'Basic', 'Test performance of electronic and mechanical systems.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.k', 'Basic', 'Investigate the chemical composition of an unknown substance.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.k', 'Basic', 'Study formation of stars using space-based telescopes.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.k', 'Basic', 'Study ways to make foods safe and healthy.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.k', 'Basic', 'Use computer models for weather forecasting.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.l', 'Basic', 'Develop medically useful products for humans.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.l', 'Basic', 'Investigate the effects of bacteria on humans and animals.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.l', 'Basic', 'Prepare environmental impact reports on wildlife.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.l', 'Basic', 'Study the science of genes.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.m', 'Basic', 'Conduct research on ways to treat a new type of illness.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.m', 'Basic', 'Evaluate the performance of a newly developed medicine.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.m', 'Basic', 'Investigate the causes of human disease.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.m', 'Basic', 'Study ways of preventing disease.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.n', 'Basic', 'Conduct research on economic issues.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.n', 'Basic', 'Study cultural differences.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.n', 'Basic', 'Study mental and emotional disorders.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.n', 'Basic', 'Study public opinion on political issues.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.o', 'Basic', 'Research and compare religious beliefs.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.o', 'Basic', 'Research influences on a literary work.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.o', 'Basic', 'Study the culture and history of a region of the world.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.o', 'Basic', 'Study the history of the family.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.p', 'Basic', 'Develop a statistical model to explain human behavior.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.p', 'Basic', 'Extend mathematical knowledge in the field of geometry.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.p', 'Basic', 'Make predictions based on statistical analyses.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.p', 'Basic', 'Solve mathematical problems.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.q', 'Basic', 'Diagnose and resolve computer software problems.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.q', 'Basic', 'Research security measures for computer systems.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.q', 'Basic', 'Test, maintain, and monitor computer programs and systems.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.q', 'Basic', 'Write and review code for software.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.r', 'Basic', 'Create a sculpture.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.r', 'Basic', 'Create computer-generated graphics or animation.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.r', 'Basic', 'Create photography art prints.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.r', 'Basic', 'Sketch a picture or design.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.s', 'Basic', 'Create appealing product designs for home appliances.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.s', 'Basic', 'Create the set for film or theater productions.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.s', 'Basic', 'Design and create handmade clothes.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.s', 'Basic', 'Design displays to sell products in stores.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.t', 'Basic', 'Act on stage for an audience.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.t', 'Basic', 'Create new dance routines.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.t', 'Basic', 'Direct stage, television, radio, video, or film productions.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.t', 'Basic', 'Study and rehearse roles from scripts.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.u', 'Basic', 'Create musical compositions, arrangements, or scores.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.u', 'Basic', 'Perform music for a live audience.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.u', 'Basic', 'Play a musical instrument.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.u', 'Basic', 'Sing in a band.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.v', 'Basic', 'Write a novel.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.v', 'Basic', 'Write advertisements.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.v', 'Basic', 'Write poetry or lyrics.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.v', 'Basic', 'Write scripts for movies and media.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.w', 'Basic', 'Develop content for a podcast.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.w', 'Basic', 'Direct a television series.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.w', 'Basic', 'Speak or read from scripted materials on radio or television.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.w', 'Basic', 'Write and report human interest stories.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.x', 'Basic', 'Coordinate activities of cooks engaged in food preparation.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.x', 'Basic', 'Create new recipes or food presentations.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.x', 'Basic', 'Develop a distinctive style of cooking.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.x', 'Basic', 'Select and ensure freshness of food and ingredients for preparation.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.y', 'Basic', 'Adapt teaching methods to meet students'' interests.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.y', 'Basic', 'Instruct students in activities designed to further intellectual growth.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.y', 'Basic', 'Teach school subjects.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.y', 'Basic', 'Teach social skills to students.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.z', 'Basic', 'Advocate for individual or community needs.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.z', 'Basic', 'Assist people with special needs.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.z', 'Basic', 'Counsel clients with mental health issues.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.z', 'Basic', 'Provide personal and economic assistance for homeless individuals.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.aa', 'Basic', 'Administer first aid treatment or life support to injured persons.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.aa', 'Basic', 'Develop medical treatment plans for patients.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.aa', 'Basic', 'Prescribe medications.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.aa', 'Basic', 'Provide preventive care to individuals.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ab', 'Basic', 'Lead a religious group.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ab', 'Basic', 'Organize religious services.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ab', 'Basic', 'Plan religious education programs.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ab', 'Basic', 'Provide spiritual guidance.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ac', 'Basic', 'Assist customers with problems or questions.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ac', 'Basic', 'Help individuals plan special events.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ac', 'Basic', 'Make reservations for guests, such as for dinner, spa treatments, or golf tee times.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ac', 'Basic', 'Make travel arrangements for sightseeing or other tours.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ad', 'Basic', 'Advise people on how to have a successful career.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ad', 'Basic', 'Coach students on study skills.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ad', 'Basic', 'Counsel people how to achieve their professional goals.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ad', 'Basic', 'Teach employees leadership and soft skills.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ae', 'Basic', 'Create business strategies or policies to increase productivity.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ae', 'Basic', 'Develop a long-term strategic plan for a company.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ae', 'Basic', 'Identify partner companies to pursue new business opportunities.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ae', 'Basic', 'Plan activities of businesses to maximize returns on investments.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.af', 'Basic', 'Demonstrate products to consumers.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.af', 'Basic', 'Direct sales or customer service activities.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.af', 'Basic', 'Persuade customers to buy a new product.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.af', 'Basic', 'Sell goods at bargain prices.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ag', 'Basic', 'Develop promotional materials for advertising.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ag', 'Basic', 'Plan an advertising or marketing campaign.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ag', 'Basic', 'Promote a new product by advertising.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ag', 'Basic', 'Use social media to market a company.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ah', 'Basic', 'Analyze financials and securities of a company.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ah', 'Basic', 'Make investments to generate future income.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ah', 'Basic', 'Manage large amounts of money for a business.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ah', 'Basic', 'Plan and direct financial decisions for a business.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ai', 'Basic', 'Keep accounting records for a company.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ai', 'Basic', 'Prepare budgets for a business.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ai', 'Basic', 'Prepare employee payroll.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ai', 'Basic', 'Verify and record numerical data for financial records.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.aj', 'Basic', 'Conduct training to improve employee skills and organizational performance.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.aj', 'Basic', 'Interpret and explain company policies and benefits to employees.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.aj', 'Basic', 'Maintain employee personnel records.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.aj', 'Basic', 'Recruit and hire employees for a company.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ak', 'Basic', 'Answer telephones, direct calls, and take messages.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ak', 'Basic', 'Maintain and update filing, inventory, and database systems.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ak', 'Basic', 'Operate office machines, such as printers and scanners.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ak', 'Basic', 'Schedule appointments for customers.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.al', 'Basic', 'Develop organizational goals or objectives.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.al', 'Basic', 'Manage an organization or group of people.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.al', 'Basic', 'Plan or coordinate one or more administrative services.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.al', 'Basic', 'Supervise organizational operations or projects.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.am', 'Basic', 'Be a speaker at a club meeting.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.am', 'Basic', 'Discuss your ideas on a podcast.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.am', 'Basic', 'Make a public speech to raise money for a worthy cause.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.am', 'Basic', 'Provide a public announcement on TV or radio.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.an', 'Basic', 'Campaign for political office.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.an', 'Basic', 'Develop knowledge of government policy decisions.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.an', 'Basic', 'Engage in a political debate.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.an', 'Basic', 'Lead a political campaign.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ao', 'Basic', 'Defend a client in court.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ao', 'Basic', 'Draw up legal documents.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ao', 'Basic', 'Mediate legal disputes.');
INSERT INTO interests_illus_activities (element_id, interest_type, activity) VALUES ('1.B.3.ao', 'Basic', 'Provide legal advice to clients.');
/*! COMMIT */;

