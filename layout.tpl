%scripts=get('scripts', [])
%scripts[:0] = ["/static/jquery-2.0.1.min.js",
%				"/static/bootstrap/js/bootstrap.min.js",
%				"/static/d3.v3.min.js"]

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title></title>
    <link rel="stylesheet" type="text/css" href="/static/bootstrap/css/bootstrap.css" media="all" />
    <link rel="stylesheet" type="text/css" href="/static/bootstrap/css/bootstrap-responsive.css" media="all" />
</head>
<body>
    <div class="container">
        %include
    </div>
%for script in scripts:
	<script src="{{script}}"></script>
%end
</body>
</html>
