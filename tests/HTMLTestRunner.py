"""
HTMLTestRunner for Python unittest framework

A TestRunner for use with the Python unit testing framework.
It generates a HTML report to show the result at a glance.
"""

import datetime
import io
import sys
import time
import unittest
from xml.sax import saxutils


# ------------------------------------------------------------------------
# The redirectors below are used to capture output during testing. Output
# sent to sys.stdout and sys.stderr are automatically captured. However
# in some cases sys.stdout is already cached before HTMLTestRunner is
# invoked (e.g. calling logging.basicConfig). In order to capture those
# output, use the redirectors for the cached stream.
#
# e.g.
#   >>> logging.basicConfig(stream=HTMLTestRunner.stdout_redirector)
#   >>>

class OutputRedirector(object):
    """ Wrapper to redirect stdout or stderr """
    def __init__(self, fp):
        self.fp = fp

    def write(self, s):
        self.fp.write(s)

    def writelines(self, lines):
        self.fp.writelines(lines)

    def flush(self):
        self.fp.flush()

stdout_redirector = OutputRedirector(sys.stdout)
stderr_redirector = OutputRedirector(sys.stderr)


# ----------------------------------------------------------------------
# Template

class Template_mixin(object):
    """
    Define a HTML template for report customerization and generation.

    Overall structure of an HTML report

    HTML
    +------------------------+
    |<html>                  |
    |  <head>                |
    |                        |
    |   STYLESHEET           |
    |   +----------------+   |
    |   |                |   |
    |   +----------------+   |
    |                        |
    |  </head>               |
    |                        |
    |  <body>                |
    |                        |
    |   HEADING              |
    |   +----------------+   |
    |   |                |   |
    |   +----------------+   |
    |                        |
    |   REPORT               |
    |   +----------------+   |
    |   |                |   |
    |   +----------------+   |
    |                        |
    |   ENDING               |
    |   +----------------+   |
    |   |                |   |
    |   +----------------+   |
    |                        |
    |  </body>               |
    |</html>                 |
    +------------------------+
    """

    STATUS = {
    0: '通过',
    1: '失败',
    2: '错误',
    }

    DEFAULT_TITLE = '单元测试报告'
    DEFAULT_DESCRIPTION = ''

    # ------------------------------------------------------------------------
    # HTML Template

    HTML_TMPL = r"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>%(title)s</title>
    <meta name="generator" content="%(generator)s"/>
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8"/>
    <style type="text/css">
        body {
            font-family: "微软雅黑", Arial, sans-serif;
            font-size: 14px;
            color: #333;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            font-size: 24px;
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        h2 {
            font-size: 18px;
            color: #333;
        }
        p {
            margin: 10px 0;
        }
        table {
            border-collapse: collapse;
            width: 100%%;
            margin-top: 20px;
            margin-bottom: 20px;
            background-color: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        th, td {
            text-align: left;
            padding: 10px;
            border: 1px solid #ddd;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .heading {
            margin-top: 20px;
            margin-bottom: 10px;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .heading .attribute {
            margin-top: 10px;
        }
        .heading .description {
            font-style: italic;
            margin-top: 10px;
            margin-bottom: 10px;
        }
        .resultTable {
            border-radius: 5px;
            overflow: hidden;
        }
        .passClass {
            background-color: #dff0d8;
        }
        .failClass {
            background-color: #f2dede;
        }
        .errorClass {
            background-color: #fcf8e3;
        }
        .passCase {
            color: #3c763d;
        }
        .failCase {
            color: #a94442;
            font-weight: bold;
        }
        .errorCase {
            color: #8a6d3b;
            font-weight: bold;
        }
        .hiddenRow {
            display: none;
        }
        .testcase {
            margin-left: 2em;
        }
        .testDescription {
            padding: 10px;
            background-color: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-top: 10px;
            margin-bottom: 10px;
        }
        .testOutput {
            padding: 10px;
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-top: 10px;
            margin-bottom: 10px;
            white-space: pre-wrap;
            font-family: monospace;
        }
        .collapse {
            cursor: pointer;
        }
        .collapse + input {
            display: none;
        }
        .collapse + input + div {
            display: none;
        }
        .collapse + input:checked + div {
            display: block;
        }
        .summary {
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
    </style>
    <script type="text/javascript">
        function showDetail(element) {
            var detailDiv = document.getElementById(element);
            if (detailDiv.style.display === "block") {
                detailDiv.style.display = "none";
            } else {
                detailDiv.style.display = "block";
            }
        }
    </script>
</head>
<body>
<h1>%(title)s</h1>
<div class="summary">
<p><strong>开始时间:</strong> %(startTime)s</p>
<p><strong>运行时长:</strong> %(duration)s</p>
<p><strong>状态:</strong> %(status)s</p>
</div>

<div class="heading">
    <p class="attribute"><strong>描述:</strong> %(description)s</p>
    <p class="attribute"><strong>总数:</strong> %(count)d</p>
    <p class="attribute"><strong>通过:</strong> %(Pass)d</p>
    <p class="attribute"><strong>失败:</strong> %(fail)d</p>
    <p class="attribute"><strong>错误:</strong> %(error)d</p>
</div>

%(report)s

</body>
</html>
"""
    REPORT_TMPL = r"""
<p>
<table class="resultTable">
<tr id="header_row">
    <th>测试类/测试方法</th>
    <th>用例数量</th>
    <th>通过</th>
    <th>失败</th>
    <th>错误</th>
    <th>查看</th>
</tr>
%(test_list)s
</table>
</p>
"""
    REPORT_CLASS_TMPL = r"""
<tr class="%(style)s">
    <td>%(desc)s</td>
    <td>%(count)s</td>
    <td>%(Pass)s</td>
    <td>%(fail)s</td>
    <td>%(error)s</td>
    <td><a href="javascript:void(0)" onclick="showDetail('%(cid)s')">详情</a></td>
</tr>
"""
    REPORT_TEST_WITH_OUTPUT_TMPL = r"""
<tr id="%(tid)s" class="%(style)s">
    <td class="testcase">%(desc)s</td>
    <td colspan="5">
        <div id="%(cid)s" class="testDetail" style="display:none;">
            <div class="testDescription">%(desc)s</div>
            <div class="testOutput">%(script)s</div>
        </div>
    </td>
</tr>
"""
    REPORT_TEST_NO_OUTPUT_TMPL = r"""
<tr id="%(tid)s" class="%(style)s">
    <td class="testcase">%(desc)s</td>
    <td colspan="5"></td>
</tr>
"""
    REPORT_TEST_OUTPUT_TMPL = r"""
%(output)s
"""

# -------------------- The end of the Template class -------------------


class _TestResult(unittest.TestResult):
    # note: _TestResult is a pure representation of results.
    # It lacks the output and reporting ability compares to unittest._TextTestResult.

    def __init__(self, verbosity=1):
        super(_TestResult, self).__init__()
        self.stdout0 = None
        self.stderr0 = None
        self.success_count = 0
        self.failure_count = 0
        self.error_count = 0
        self.verbosity = verbosity

        # result is a list of result in 4 tuple
        # (
        #   result code (0: success; 1: fail; 2: error),
        #   TestCase object,
        #   Test output (byte string),
        #   stack trace,
        # )
        self.result = []

    def startTest(self, test):
        super(_TestResult, self).startTest(test)
        # just one buffer for both stdout and stderr
        self.outputBuffer = io.StringIO()
        stdout_redirector.fp = self.outputBuffer
        stderr_redirector.fp = self.outputBuffer
        self.stdout0 = sys.stdout
        self.stderr0 = sys.stderr
        sys.stdout = stdout_redirector
        sys.stderr = stderr_redirector

    def complete_output(self):
        """
        Disconnect output redirection and return buffer.
        Safe to call multiple times.
        """
        if self.stdout0:
            sys.stdout = self.stdout0
            sys.stderr = self.stderr0
            self.stdout0 = None
            self.stderr0 = None
        return self.outputBuffer.getvalue()

    def stopTest(self, test):
        # Usually one of addSuccess, addError or addFailure would have been called.
        # But there are some path in unittest that would bypass this.
        # We must disconnect stdout in stopTest(), which is guaranteed to be called.
        self.complete_output()

    def addSuccess(self, test):
        self.success_count += 1
        super(_TestResult, self).addSuccess(test)
        output = self.complete_output()
        self.result.append((0, test, output, ''))
        if self.verbosity > 1:
            sys.stderr.write('ok ')
            sys.stderr.write(str(test))
            sys.stderr.write('\n')
        else:
            sys.stderr.write('.')

    def addError(self, test, err):
        self.error_count += 1
        super(_TestResult, self).addError(test, err)
        _, _exc_str = self.errors[-1]
        output = self.complete_output()
        self.result.append((2, test, output, _exc_str))
        if self.verbosity > 1:
            sys.stderr.write('E  ')
            sys.stderr.write(str(test))
            sys.stderr.write('\n')
        else:
            sys.stderr.write('E')

    def addFailure(self, test, err):
        self.failure_count += 1
        super(_TestResult, self).addFailure(test, err)
        _, _exc_str = self.failures[-1]
        output = self.complete_output()
        self.result.append((1, test, output, _exc_str))
        if self.verbosity > 1:
            sys.stderr.write('F  ')
            sys.stderr.write(str(test))
            sys.stderr.write('\n')
        else:
            sys.stderr.write('F')


class HTMLTestRunner(Template_mixin):
    """
    """
    def __init__(self, stream=sys.stdout, verbosity=1, title=None, description=None):
        self.stream = stream
        self.verbosity = verbosity
        if title is None:
            self.title = self.DEFAULT_TITLE
        else:
            self.title = title
        if description is None:
            self.description = self.DEFAULT_DESCRIPTION
        else:
            self.description = description

    def run(self, test):
        "Run the given test case or test suite."
        result = _TestResult(self.verbosity)
        test(result)
        self.stopTime = datetime.datetime.now()
        self.generateReport(test, result)
        return result

    def sortResult(self, result_list):
        # unittest does not seems to run in any particular order.
        # Here at least we want to group them together by class.
        rmap = {}
        classes = []
        for n,t,o,e in result_list:
            cls = t.__class__
            if not cls in rmap:
                rmap[cls] = []
                classes.append(cls)
            rmap[cls].append((n,t,o,e))
        r = [(cls, rmap[cls]) for cls in classes]
        return r

    def getReportAttributes(self, result):
        """
        Return report attributes as a list of (name, value).
        Override this to add custom attributes.
        """
        startTime = str(self.startTime)[:19]
        duration = str(self.stopTime - self.startTime)
        status = []
        if result.success_count: status.append('通过 %s' % result.success_count)
        if result.failure_count: status.append('失败 %s' % result.failure_count)
        if result.error_count:   status.append('错误 %s' % result.error_count  )
        if status:
            status = ' '.join(status)
        else:
            status = '无结果'
        return [
            ('开始时间', startTime),
            ('运行时长', duration),
            ('状态', status),
        ]

    def generateReport(self, test, result):
        self.startTime = datetime.datetime.now()
        report_attrs = self.getReportAttributes(result)
        generator = 'HTMLTestRunner %s' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stylesheet = self._generate_stylesheet()
        heading = self._generate_heading(report_attrs)
        report = self._generate_report(result)
        ending = self._generate_ending()
        output = self.HTML_TMPL % dict(
            title = saxutils.escape(self.title),
            generator = generator,
            stylesheet = stylesheet,
            heading = heading,
            report = report,
            ending = ending,
            startTime = report_attrs[0][1],
            duration = report_attrs[1][1],
            status = report_attrs[2][1],
            description = saxutils.escape(self.description),
            count = str(result.success_count + result.failure_count + result.error_count),
            Pass = str(result.success_count),
            fail = str(result.failure_count),
            error = str(result.error_count),
        )
        self.stream.write(output.encode('utf8'))

    def _generate_stylesheet(self):
        return self.STYLESHEET_TMPL

    def _generate_heading(self, report_attrs):
        a_lines = []
        for name, value in report_attrs:
            line = self.HEADING_ATTRIBUTE_TMPL % dict(
                    name = saxutils.escape(name),
                    value = saxutils.escape(value),
                )
            a_lines.append(line)
        heading = self.HEADING_TMPL % dict(
            title = saxutils.escape(self.title),
            parameters = ''.join(a_lines),
            description = saxutils.escape(self.description),
        )
        return heading

    def _generate_report(self, result):
        rows = []
        sortedResult = self.sortResult(result.result)
        for cid, (cls, cls_results) in enumerate(sortedResult):
            # subtotal for a class
            np = nf = ne = 0
            for n,t,o,e in cls_results:
                if n == 0: np += 1
                elif n == 1: nf += 1
                else: ne += 1

            # format class description
            if cls.__module__ == "__main__":
                name = cls.__name__
            else:
                name = "%s.%s" % (cls.__module__, cls.__name__)
            doc = cls.__doc__ and cls.__doc__.split("\n")[0] or ""
            desc = doc and '%s: %s' % (name, doc) or name

            row = self.REPORT_CLASS_TMPL % dict(
                style = ne > 0 and 'errorClass' or nf > 0 and 'failClass' or 'passClass',
                desc = desc,
                count = np+nf+ne,
                Pass = np,
                fail = nf,
                error = ne,
                cid = 'c%s' % (cid+1),
            )
            rows.append(row)

            for tid, (n,t,o,e) in enumerate(cls_results):
                self._generate_report_test(rows, cid, tid, n, t, o, e)

        report = self.REPORT_TMPL % dict(
            test_list = ''.join(rows),
        )
        return report

    def _generate_report_test(self, rows, cid, tid, n, t, o, e):
        # e.g. 'pt1.1', 'ft1.1', etc
        has_output = bool(o or e)
        tid = (n == 0 and 'p' or 'f') + 't%s.%s' % (cid+1,tid+1)
        name = t.id().split('.')[-1]
        doc = t.shortDescription() or ""
        desc = doc and ('%s: %s' % (name, doc)) or name
        tmpl = has_output and self.REPORT_TEST_WITH_OUTPUT_TMPL or self.REPORT_TEST_NO_OUTPUT_TMPL

        # o and e should be byte string because they are collected from stdout and stderr?
        if isinstance(o,str):
            # uo = unicode(o.encode('string_escape'))
            uo = o
        else:
            uo = o
        if isinstance(e,str):
            # ue = unicode(e.encode('string_escape'))
            ue = e
        else:
            ue = e

        script = self.REPORT_TEST_OUTPUT_TMPL % dict(
            output = saxutils.escape(uo+ue),
        )

        row = tmpl % dict(
            tid = tid,
            style = n == 0 and 'passCase' or (n == 1 and 'failCase' or 'errorCase'),
            desc = desc,
            script = script,
            cid = 'c%s' % (cid+1),
        )
        rows.append(row)
        if not has_output:
            return

    def _generate_ending(self):
        return self.ENDING_TMPL 