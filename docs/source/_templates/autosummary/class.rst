{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% set ns_methods = namespace(items=[]) %}
   {% for item in methods %}
      {% if item not in inherited_members and item != '__init__' %}
         {% set ns_methods.items = ns_methods.items + [item] %}
      {% endif %}
   {% endfor %}
   {% if ns_methods.items %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in ns_methods.items %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% set ns_attrs = namespace(items=[]) %}
   {% for item in attributes %}
      {% if item not in inherited_members %}
         {% set ns_attrs.items = ns_attrs.items + [item] %}
      {% endif %}
   {% endfor %}
   {% if ns_attrs.items %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in ns_attrs.items %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

